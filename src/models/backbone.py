import torch
import torch.nn as nn

from reference_scan import selective_scan_ref
from src.models.ssm_block import SSMBlock


class RMSNorm(nn.Module):
    """
    No mean-centering, no bias
    Cheaper than LayerNorm
    Standard in Mamba stacks
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.pow(2).mean(dim = -1, keepdim = True).add(self.eps).rsqrt()
        return x * norm * self.weight


class SSMBackbone(nn.Module):
    def __init__(
            self,
            d_model,
            d_state,
            n_layers,
            d_conv = 4,
            expand = 2,
            selective = True,
            discretization = 'zoh',
            pooling = 'mean',
            n_mels = 64,
            learned_input_embed = False,
    ):
        super().__init__()
        self.pooling = pooling

        self.blocks = nn.ModuleList([
            SSMBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                selective=selective,
                discretization=discretization,
            )
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)

    def forward(self, x, mode = 'sequence'):
        """
        x: (batch, T, d_model) — d_model == n_mels here, since
           model.learned_input_embed is False (no input embedding layer).
           GAP 3, only relevant if that flag ever flips true: add an
           nn.Linear(n_mels, d_model) before the block loop.
        mode: "sequence" -> (batch, T, d_model). The prediction head (1.5)
              attaches here — pooling collapses time, so there's no "next
              frame" left once you pool.
              "pooled" -> (batch, d_model). What Option 3's distance head
              consumes at inference.
        """
        for block, norm in zip(self.blocks, self.norms):
            x = x + block(norm(x)) # pre-norm residual

        x = self.final_norm(x)

        if mode == 'sequence':
            return x
        elif mode == 'pooled':
            return self._pool(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _pool(self, x): # (batch, T, d_model)
        if self.pooling == 'mean':
            # better for anomalies that are steady hums
            return x.mean(dim = 1)
        elif self.pooling == 'last':
            # better for anomalies that are steady hums, worse than mean
            return x[:, -1, :]
        elif self.pooling == 'max':
            # better for anomalies that are transient clicks
            return x.max(dim = 1).values
        elif self.pooling == 'concat_mean_last':
            return torch.cat([x.mean(dim = 1), x[:, -1, :]], dim = -1) # doubles d_model downstream
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")