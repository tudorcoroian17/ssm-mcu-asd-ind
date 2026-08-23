import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv: int = 4, expand: int = 2, selective: bool = True,
                 discretization: str = 'zoh', dt_rank=None):
        super().__init__()

        if dt_rank is None:
            dt_rank = max(1, d_model // 16)

        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.selective = selective
        self.discretization = discretization

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True
        )

        if selective:
            # Dependent on input
            self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False) # Contains delta, B, and C
            self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        else:
            # Static
            self.B = nn.Parameter(torch.randn(d_state))
            self.C = nn.Parameter(torch.randn(d_state))
            self.dt = nn.Parameter(torch.zeros(self.d_inner))

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0).repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def discretize(self, delta, A, B):
        deltaA = delta.unsqueeze(-1) * A # (..., d_inner, d_state)
        if self.discretization == "zoh":
            A_bar = torch.exp(deltaA)
            exact_coef = torch.expm1(deltaA) / A # expm1(x) can also be approximated by x + (x^2)/2 (Taylor series)
            coef = torch.where(deltaA.abs() < 1e-4, delta.unsqueeze(-1), exact_coef)
            B_bar = coef * B.unsqueeze(-2)
        elif self.discretization == "euler":
            deltaA_stable = torch.clamp(deltaA, min=-1.9) # need to clamp delta * A above -2
            A_bar = 1 + deltaA_stable
            B_bar = delta.unsqueeze(-1) * B.unsqueeze(-2)
        else:
            raise ValueError(f'Unknown discretization: {self.discretization}')

        return A_bar, B_bar

    def _scan(self, A_bar, B_bar, C, u, return_h_trace=False):
        batch, T = u.shape[0], u.shape[1]
        h = torch.zeros(batch, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        h_trace = [] if return_h_trace else None

        A_bar_ts = A_bar.unbind(dim=1)
        B_bar_ts = B_bar.unbind(dim=1)
        u_ts = u.unbind(dim=1)
        if self.selective:
            C_ts = C.unbind(dim=1)

        ys = []
        if self.selective:
            for t in range(T):
                Bu_t = B_bar_ts[t] * u_ts[t].unsqueeze(-1)
                h = torch.addcmul(Bu_t, A_bar_ts[t], h)  # Bu_t + A_bar_ts[t] * h, one kernel
                if return_h_trace:
                    h_trace.append(h.detach().abs().max().item())
                ys.append((h * C_ts[t].unsqueeze(1)).sum(dim=-1))
        else:
            for t in range(T):
                Bu_t = B_bar_ts[t] * u_ts[t].unsqueeze(-1)
                h = torch.addcmul(Bu_t, A_bar_ts[t], h)
                if return_h_trace:
                    h_trace.append(h.detach().abs().max().item())
                ys.append((h * C).sum(dim=-1))

        y = torch.stack(ys, dim=1)
        return (y, h_trace) if return_h_trace else y

    def forward(self, x, return_h_trace=False):
        # return_h_trace is just a debug hook
        batch, T, _ = x.shape

        # split into working signal (u) and gate (z)
        xz = self.in_proj(x) # (batch, T, 2*d_inner)
        u, z = xz.chunk(2, dim=-1) # each (batch, T, d_inner)

        # causal depthwise conv, trimmed to first T
        u = u.transpose(1, 2) # (batch, d_inner, T)
        u = self.conv(u)[:, :, :T] # grab all values on dim1 and dim2; grab all values up to (not including) T on dim3
        u = u.transpose(1, 2) # (batch, T, d_inner)
        u = F.silu(u) # add non-linearity so in_proj + conv + x_proj don't collapse into one linear map

        if self.selective:
            x_dbl = self.x_proj(u) # (batch, T, dt_rank + 2 * d_inner) -> dim 3 has delta, B, and C
            delta_low, B, C = torch.split(
                x_dbl, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1
            )
            delta_raw = self.dt_proj(delta_low) # (batch, T, d_inner)
        else:
            delta_raw = self.dt # (d_inner) -> broadcasts over (batch, T)
            B = self.B # (d_inner)
            C = self.C # (d_inner)

        delta = F.softplus(delta_raw) # force delta positive for stability guarantee

        A = -torch.exp(self.A_log) # (d_inner, d_state)

        # Materialize A_bar, B_bar at full to take advantage of GPU parallelism
        # On MCU, should only materialize A_bar_t, B_bar_t (per timestep)
        A_bar, B_bar = self.discretize(delta, A, B) # (batch, T, d_inner, d_state)

        # fixed branch never gets a batch/T dimension from discretize()
        # (delta/A/B have none to broadcast against), but _scan()'s A_bar[:, t]
        # indexing assumes one. .expand() is a view -- no extra memory allocated.
        if not self.selective:
            A_bar = A_bar.unsqueeze(0).unsqueeze(0).expand(batch, T, -1, -1)
            B_bar = B_bar.unsqueeze(0).unsqueeze(0).expand(batch, T, -1, -1)

        # the scan itself; sequential, one frame at a time
        result = self._scan(A_bar, B_bar, C, u, return_h_trace)
        y, h_trace = result if return_h_trace else (result, None)

        # raw shortcut (D), gate (z), project back to d_model
        y = y + u * self.D
        y = y * F.silu(z)
        out = self.out_proj(y)
        return (out, h_trace) if return_h_trace else out # (batch, T, d_model)