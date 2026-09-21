# Quantization schemes in the MCU deployment matrix: arithmetic, storage, and C implementation

**Feeds:** `findings/150_ranges_and_activations_for_quantization.md` (origin of
`ranges.json`); `findings/520` to `findings/523` (Python validation of the
weight and activation schemes); `findings/530` (per-access weight
dequantization); `findings/540` to `findings/543` (deployment results, the true
int8 C port, and the int8 head); `mcu/export_deploy_matrix.py`;
`mcu/export_deploy_matrix_classic.py`.

## Result

The deployment matrix contains four backbone schemes and two head schemes. The
classic (`selective=False`) branch adds one more axis, the recurrence form.
The schemes differ in one main way: where the arithmetic happens.

- **Weight-only int8** and **weight + activation-boundary int8** store int8
  values but compute in fp32. Every int8 weight is converted to float at the
  moment the C code reads it. These two schemes are fake quantization.
- **True int8** stores int8 values and computes with int32 accumulators. Every
  matrix product, the depthwise convolution, and the state update use integer
  multiply-accumulate. The code converts to float to rescale and to compute
  `A_bar`, and it uses look-up tables (LUTs) for SiLU and softplus.

One `ssm_backbone.c` source serves all fake-quant schemes of a branch. The
generated `ssm_weights.h` decides what each read means. A second source,
`ssm_true_int8_src/ssm_backbone.c`, serves both true int8 h-widths. This design
keeps the `.c` text identical across setups that share a source.

Table 1 summarizes the schemes.

| Scheme | Stored in flash | Arithmetic in the backbone | Activations | Head |
| :--- | :--- | :--- | :--- | :--- |
| Full fp32 | fp32 tensors | fp32 | fp32 | fp32 |
| Weight-only int8 | int8 tensors + fp32 scale(s) | fp32, dequantize on each read | fp32 | fp32 |
| Weight + activation-boundary int8 | Same as weight-only, plus 4 scales per layer and 1 final-norm scale | fp32, plus quantize-dequantize at 5 points | int8 grid, held in fp32 | fp32 |
| True int8 | int8 weights, per-tensor scales, LUTs; fp32 `A`, `D`, biases, norm weights | int32 accumulate, float rescale | int8 (`h`: int8 or int16) | fp32 or int8 |

## Scope and sources

This finding covers the backbone and head quantization in `mcu/`. It does not
cover the log-mel feature stage (see `findings/573` and `findings/574` for the
RP2040 q15 path) or the threshold-fitting methods (see `findings/140` and
`findings/543`).

The finding is based on these files, all read without modification:

- `mcu/export_deploy_matrix.py` (lines 1 to 1000 of 1366) and
  `mcu/export_deploy_matrix_classic.py` (all 988 lines).
- `mcu/ssm_backbone_src/`, `mcu/ssm_true_int8_src/`,
  `mcu/ssm_backbone_classic_src/`, `mcu/ssm_true_int8_classic_src/`.
- `checks/smoke/quant/weight_quant_parity.py`, `activation_quant.py`,
  `activation_quant_parity.py`, `true_int8_sim.py`, and
  `true_int8_sim_classic.py` (lines 1 to 260 of 405).
- Generated headers under `mcu/deploy/case1/086acf0275b8/setups/` (selective)
  and `mcu/deploy/case1/b77482e85dc3/setups/` (classic).

Code excerpts keep the source text. Long comments are shortened and marked
`/* ... */`. Each excerpt names its file and function.

## Shared building blocks

### Symmetric int8 quantization

Every scheme uses the same symmetric mapping. The range is $[-127, 127]$. The
value $-128$ is never produced.

$$q = \operatorname{clip}\left(\operatorname{round}\left(\frac{x}{s}\right), -127, 127\right), \qquad \hat{x} = q \cdot s$$

The scale $s$ has three sources (Table 2).

| Tensor kind | Scale formula | Where it is computed |
| :--- | :--- | :--- |
| Weight, per-tensor | $s = \max\lvert w \rvert / 127$ | `fq_quantize_int8_pertensor` in `export_deploy_matrix.py` |
| Weight, per-channel | $s_r = \max_c \lvert w_{r,c} \rvert / 127$, one scale per output row $r$ | `fq_quantize_int8_perchannel` in `export_deploy_matrix.py` |
| Activation, from `ranges.json` | $s = \max(\lvert\min\rvert, \lvert\max\rvert) / 127$ | `load_ranges_scales` in `activation_quant_parity.py` |
| Activation, from hooks (true int8 only) | $s = \max_{\text{clips}} \max\lvert a \rvert / 127$ over 8 training clips | `calibrate_missing_ranges` in `true_int8_sim.py` |
| `h`, int16 width | $s_h = s_h^{(127)} \cdot 127 / 32767$ | `prepare_quantized_model` in `true_int8_sim.py` |

The weight functions, as they appear in `export_deploy_matrix.py`:

```python
def fq_quantize_int8_pertensor(w_np):
    max_abs = float(np.max(np.abs(w_np)))
    scale = max_abs / 127.0 if max_abs > 0.0 else 1e-12
    q = np.clip(np.round(w_np / scale), -127, 127).astype(np.int8)
    return q, np.array([scale], dtype=np.float32)


def fq_quantize_int8_perchannel(w_np):
    reduce_axes = tuple(range(1, w_np.ndim))
    max_abs = np.max(np.abs(w_np), axis=reduce_axes)
    scale = np.where(max_abs == 0.0, 1e-12, max_abs / 127.0).astype(np.float32)
    scale_bc = scale.reshape((-1,) + (1,) * (w_np.ndim - 1))
    q = np.clip(np.round(w_np / scale_bc), -127, 127).astype(np.int8)
    return q, scale
```

The activation scale function, as it appears in `activation_quant_parity.py`
(docstring shortened):

```python
def load_ranges_scales(base_dir):
    with open(base_dir / 'ranges.json') as f:
        ranges = json.load(f)
    scales = {}
    for name, stats in ranges.items():
        max_abs = max(abs(stats['min']), abs(stats['max']))
        scales[name] = max_abs / 127.0 if max_abs > 0 else 1e-12
    return scales
```

### Illustrative example: per-tensor and per-channel

The values below are invented for this explanation. They are not project data.

A weight matrix has two rows: row 0 is `[0.50, -1.27]` and row 1 is
`[0.0203, 0.0127]`.

- **Per-tensor.** The largest magnitude is 1.27, so $s = 0.01$. Row 1 becomes
  `[2, 1]`. Dequantized, it reads `[0.02, 0.01]`. The value 0.0127 has an error
  of 0.0027, which is 21 percent.
- **Per-channel.** Row 0 keeps $s_0 = 0.01$. Row 1 gets $s_1 = 0.0203 / 127
  \approx 0.00015984$ and becomes `[127, 79]`. Dequantized, 0.0127 reads
  0.012628. The error is 0.000072, which is 0.6 percent.

Per-channel scaling costs one extra fp32 value per output row. It protects rows
with a small dynamic range from rows with a large one.

## Scheme 1: weight-only int8 (fake quantization)

### What is quantized

The exporter uses two weight modes. The mode selects tensors by name suffix in
`WEIGHT_MODE_SUFFIXES` (`weight_quant_parity.py`).

| Mode | Tensors quantized (selective model) | Tensors left in fp32 |
| :--- | :--- | :--- |
| `projections` | `in_proj.weight`, `conv.weight`, `x_proj.weight`, `dt_proj.weight`, `out_proj.weight`, all RMSNorm weights | `conv.bias`, `dt_proj.bias`, `D`, `A` |
| `all` | Everything in `projections`, plus `conv.bias`, `dt_proj.bias`, `D`, `A_log` | None |
| `none` (baseline) | None | Everything |

Two rules apply to all modes:

- One-dimensional tensors (biases, `D`, norm weights) always use per-tensor
  scaling, even in a per-channel setup. A per-channel scale on a one-dimensional
  tensor would need one scale per element.
- `A` is a special case. The checkpoint stores `A_log`. Quantization acts on
  `A_log`, not on `A = -exp(A_log)`. This follows `findings/521`.

The test that decides each tensor is one line:

```python
def should_quantize(param_name, mode='projections'):
    return any(s in param_name for s in WEIGHT_MODE_SUFFIXES[mode])
```

On the classic branch, only `all` exists. `classic_should_quantize` in
`export_deploy_matrix_classic.py` adds one more rule: under `qabar`, the tensors
`.A_log`, `.dt`, and `.B` do not ship. Section "Classic branch: the recurrence
axis" explains why.

### Storage in the generated header

The exporter writes one int8 array and one scale array per quantized tensor. It
also writes a macro that dequantizes on access. The macro is the only place
that knows the scheme.

Per-channel (`w_proj_perchannel_knn16_fp32/ssm_weights.h`). The scale index is
the row `(r)`:

```c
#define SSM_IN_PROJ_W(w, r, c) ((float)(w)->in_proj_w_q[(r) * SSM_D_MODEL + (c)] * (w)->in_proj_w_scale[(r)])
#define SSM_CONV_W(w, r, c) ((float)(w)->conv_w_q[(r) * SSM_D_CONV + (c)] * (w)->conv_w_scale[(r)])
```

Per-tensor (`w_all_pertensor_actboundaries_knn16_fp32/ssm_weights.h`). The
scale index is always `0`:

```c
#define SSM_IN_PROJ_W(w, r, c) ((float)(w)->in_proj_w_q[(r) * SSM_D_MODEL + (c)] * (w)->in_proj_w_scale[0])
#define SSM_CONV_B(w, i) ((float)(w)->conv_b_q[(i)] * (w)->conv_b_scale[0])
```

`A` under `all` stores `A_log` as int8 and applies `-expf` on every access:

```c
#define SSM_A(w, r, c) (-expf((float)(w)->A_log_q[(r) * SSM_D_STATE + (c)] * (w)->A_log_scale[0]))
```

`A` under `projections` and under the fp32 baseline is precomputed at export
time. The header then holds a plain float read:

```c
#define SSM_A(w, r, c) ((w)->A[(r) * SSM_D_STATE + (c)])
```

The exporter code that produces these macros is `fq_emit_2d_field`. The
quantized branch, as it appears in `export_deploy_matrix.py`:

```python
    use_perchannel = granularity == "per-channel" and flat.ndim >= 2
    q, scale = (fq_quantize_int8_perchannel(flat) if use_perchannel else fq_quantize_int8_pertensor(flat))
    decl_lines.append(c_int8_array(f"{field_name}_q", q))
    decl_lines.append(c_array(f"{field_name}_scale", scale))
    member_decls.append(f"    const int8_t *{bare}_q;")
    member_decls.append(f"    const float *{bare}_scale;")
    scale_idx = "(r)" if use_perchannel else "0"
    macro_lines.append(
        f"#define {macro_name}(w, r, c) "
        f"((float)(w)->{bare}_q[(r) * {ncols_macro} + (c)] * (w)->{bare}_scale[{scale_idx}])")
```

### Arithmetic in the backbone

The backbone source does not change. It reads weights through the macros and
multiplies in fp32. This excerpt is the `in_proj` section of
`ssm_block_step` in `mcu/ssm_backbone_src/ssm_backbone.c`:

```c
	for (int o = 0; o < SSM_D_INNER; o++) {
		float acc = 0.0f;
		for (int i = 0; i < SSM_D_MODEL; i++) {
			acc += SSM_IN_PROJ_W(w, o, i) * x_norm[i];
		}
		u_raw[o] = acc;
	}
```

Each inner-loop step performs one int8-to-float conversion, one float multiply
by the scale, and one float multiply-accumulate. The scheme saves flash. It
does not save compute.

## Scheme 2: weight + activation-boundary int8 (fake quantization)

This scheme adds activation quantization to weight scheme `all` with
per-tensor scales. The exporter builds only this combination. No setup pairs
activation quantization with `projections` (see the `activation_group` entries
in `build_setup_matrix`).

### Where activations are quantized

The scheme quantizes five points, called the boundaries group
(`ACTIVATION_GROUP_SUFFIXES['boundaries']` in `activation_quant.py`):

| C macro | `ranges.json` key | Point in the block |
| :--- | :--- | :--- |
| `SSM_QUANT_U` | `block<i>.u_post_conv_silu` | After the depthwise convolution and SiLU |
| `SSM_QUANT_Z` | `block<i>.z_gate` | After the `z` half of `in_proj` |
| `SSM_QUANT_Y_GATED` | `block<i>.y_gated` | After the D shortcut and the SiLU gate |
| `SSM_QUANT_BLOCK_OUT` | `block<i>.block_output` | After `out_proj` |
| `SSM_QUANT_FINAL_NORM` | `final_norm_output` | After the final RMSNorm |

The recurrence internals (`delta`, `A_bar`, `B_bar`, `h`, `B`, `C`, `y_scan`)
belong to the `scan` group. They stay in fp32 in this scheme. The source says
so directly: `y_c` has no quantize hook because it is a scalar consumed on the
line where it is produced.

### Implementation

The quantize-dequantize function is one small routine in
`mcu/ssm_backbone_src/ssm_backbone.c`:

```c
static inline float ssm_quant_dequant(float v, float scale) {
	float q = roundf(v / scale);
	if (q > 127.0f) q = 127.0f;
	if (q < -127.0f) q = -127.0f;
	return q * scale;
}
```

The generated header binds each macro either to this function or to a
pass-through. With boundaries on (`w_all_pertensor_actboundaries_knn16_fp32`):

```c
#define SSM_QUANT_U(w, val) (ssm_quant_dequant((val), (w)->u_scale))
#define SSM_QUANT_Z(w, val) (ssm_quant_dequant((val), (w)->z_scale))
#define SSM_QUANT_Y_GATED(w, val) (ssm_quant_dequant((val), (w)->y_gated_scale))
#define SSM_QUANT_BLOCK_OUT(w, val) (ssm_quant_dequant((val), (w)->block_out_scale))
#define SSM_QUANT_FINAL_NORM(val) (ssm_quant_dequant((val), ssm_final_norm_scale))
```

With boundaries off (`w_proj_perchannel_knn16_fp32`), the same macros do
nothing:

```c
#define SSM_QUANT_U(w, val) (val)
#define SSM_QUANT_Z(w, val) (val)
```

The backbone calls each macro once, after the tensor is complete. For
example, the `z` path and the final norm:

```c
	for (int o = 0; o < SSM_D_INNER; o++) {
		z[o] = SSM_QUANT_Z(w, z[o]);
	}
```

```c
	float normed[SSM_D_MODEL];
	ssm_rmsnorm(x, &ssm_final_norm_w, normed);
	for (int i = 0; i < SSM_D_MODEL; i++) {
		normed[i] = SSM_QUANT_FINAL_NORM(normed[i]);
	}
```

The activation scales are static. The exporter writes them into the per-layer
struct as plain floats, in the same loop iteration as their struct members.
This ordering matters: the exporter uses positional struct initializers, so a
member declaration and its initializer value must be appended together
(see the warning above `FQ_ACTIVATION_GROUPS` in `export_deploy_matrix.py`).

The pooled embedding stays in fp32 here. `SSMBackbone_GetPooled` returns the
mean of the per-frame values. It does not requantize the mean. The true int8
scheme differs on this point (Scheme 3).

## Scheme 3: true int8

### Data types and state

The source is `mcu/ssm_true_int8_src/ssm_backbone.{h,c}`. One `#define` in the
generated `ssm_weights.h` selects the `h` storage width. The header
`ssm_backbone.h` reads it:

```c
#ifdef SSM_H_WIDTH_INT16
typedef int16_t ssm_h_t;
#define SSM_H_CLIP 32767
#else
typedef int8_t ssm_h_t;
#define SSM_H_CLIP 127
#endif
```

The state structure uses int8 or int16 for everything that persists between
frames, except the pooling sum:

```c
typedef struct {
	ssm_h_t h[SSM_N_LAYERS][SSM_D_INNER][SSM_D_STATE];
	int8_t conv_hist_q[SSM_N_LAYERS][SSM_D_INNER][SSM_D_CONV - 1]; /* oldest first */
	float pooled_sum[SSM_D_MODEL];
	uint32_t frame_count;
} SSMBackbone_State;
```

For the model in `086acf0275b8` (2 layers, `SSM_D_INNER` 128, `SSM_D_STATE` 8,
`SSM_D_CONV` 4), `h` holds 2,048 elements. That is 8,192 bytes in fp32, 4,096
bytes at int16, and 2,048 bytes at int8. `conv_hist_q` holds 768 elements, or
768 bytes.

`conv_hist_q` always uses int8. It stores `u_raw_q`, which is quantized to the
`s_conv_out` domain and never to the `h` domain.

### Quantization primitives

These four functions are the whole integer toolkit. They are in
`ssm_true_int8_src/ssm_backbone.c`:

```c
static inline int8_t ssm_requantize(float real, float scale) {
	float q = roundf(real / scale);
	if (q > 127.0f) q = 127.0f;
	if (q < -127.0f) q = -127.0f;
	return (int8_t) q;
}

static void ssm_quantize_arr(const float *real, float scale, int n, int8_t *out_q) {
	for (int i = 0; i < n; i++) {
		out_q[i] = ssm_requantize(real[i], scale);
	}
}

static inline int8_t ssm_apply_lut(const int8_t *lut, int8_t x_q) {
	return lut[(int) x_q + 128];
}

static void ssm_qlinear_real(const int8_t *x_q, const int8_t *w_q,
		int out_dim, int in_dim, float x_scale, float w_scale,
		const float *bias, float *out_real) {
	for (int o = 0; o < out_dim; o++) {
		const int8_t *w_row = &w_q[o * in_dim];
		int32_t acc = 0;
		for (int i = 0; i < in_dim; i++) {
			acc += (int32_t) w_row[i] * (int32_t) x_q[i];
		}
		float real = (float) acc * x_scale * w_scale;
		if (bias != NULL) {
			real += bias[o];
		}
		out_real[o] = real;
	}
}
```

`ssm_qlinear_real` shows the pattern of every integer operation:

1. Multiply two int8 values as int32 and accumulate in an int32.
2. Convert the accumulator to float and multiply by the two scales. The result
   is in real units.
3. Add the bias in real units. Biases are not quantized to int32 in this scheme.
4. The caller then calls `ssm_requantize` to return to int8 at the next
   tensor's scale.

In equation form, for one output element:

$$y_o = s_x \, s_w \sum_i w_{o,i}\, x_i + b_o, \qquad q_o = \operatorname{clip}\left(\operatorname{round}\left(\frac{y_o}{s_y}\right), -127, 127\right)$$

### Tensors and their scales

Table 3 lists each tensor in one block and the scale that governs it. The
scale names match the fields of `ssm_true_int8_layer_t`.

| Tensor | Type | Scale field | Origin of the scale |
| :--- | :--- | :--- | :--- |
| `x_norm_q` (RMSNorm output) | int8 | `s_norm_out` | Hook on `norms[i]` |
| `u_raw_q` (`in_proj`, u half) | int8 | `s_conv_out` | Hook on `conv` output |
| `z_q` (`in_proj`, z half) | int8 | `s_z_gate` | `ranges.json` |
| `conv_out_q` | int8 | `s_conv_out` | Hook on `conv` output |
| `u_q` (after SiLU LUT) | int8 | `s_u_post_conv_silu` | `ranges.json` |
| `delta_low_q` | int8 | `s_x_proj_delta_low_out` | Hook on `x_proj` |
| `B_q`, `C_q` | int8 | `s_B`, `s_C` | `ranges.json` |
| `delta_raw_q` | int8 | `s_dt_proj_out` | Hook on `dt_proj` |
| `delta_q` (after softplus LUT) | int8 | `s_delta` | `ranges.json` |
| `A_bar_q` | int8 | `s_A_bar` | `ranges.json` |
| `B_bar_q` | int8 | `s_B_bar` | `ranges.json` |
| `h` | int8 or int16 | `s_h` | `ranges.json`, rescaled for int16 |
| `y_scan_q` | int8 | `s_y_scan` | `ranges.json` |
| `silu_z_q` (after LUT) | int8 | `s_silu_z` | Range of the SiLU function itself |
| `y_gated_q` | int8 | `s_y_gated` | `ranges.json` |
| `block_out_q` | int8 | `s_block_output` | `ranges.json` |
| `normed_q` (final norm) | int8 | `ssm_final_norm_scale` | `ranges.json` |

### Weights

All large weight tensors use one per-tensor scale each. The `in_proj` matrix
has one shared scale. The exporter quantizes the whole matrix, then splits it
into a u half and a z half. Both halves use `in_proj_w_scale`.

`A` and `D` are the exception. Python quantizes `A_log` and `D` per-tensor,
then converts them back to real values at export time:

```python
    A_log_q, A_log_scale = quantize_weight_pertensor(sd[f"blocks.{i}.A_log"].numpy())
    w["A_real"] = -np.exp(A_log_q * A_log_scale)  # static per layer, precomputed once
    D_q, D_scale = quantize_weight_pertensor(sd[f"blocks.{i}.D"].numpy())
    w["D_real"] = D_q * D_scale  # static per layer, precomputed once
```

The C code receives `A` and `D` as fp32 arrays (`const float *A; const float *D;`
in `ssm_true_int8_layer_t`). The values carry int8 rounding noise, but they
occupy 4 bytes per element in flash. Biases (`conv_b`, `dt_proj_b`) and RMSNorm
weights (`norm_w`) also stay fp32.

### Look-up tables

SiLU and softplus use 256-entry int8 LUTs. The index is the int8 input plus
128. The exporter builds each LUT in `build_lut` (`true_int8_sim.py`):

```python
def build_lut(fn, in_scale, out_scale=None):
    raw = np.array([fn(i * in_scale) for i in range(-128, 128)])
    if out_scale is None:
        max_abs = float(np.max(np.abs(raw)))
        out_scale = max_abs / 127.0 if max_abs > 0.0 else 1e-12
    lut = np.clip(np.round(raw / out_scale), -127, 127).astype(np.int32)
    return lut, out_scale
```

`prepare_quantized_model` builds three LUTs per layer:

```python
        lut_softplus, _ = build_lut(softplus_fp32, s["dt_proj_out"], s["delta"])
        lut_silu_conv, _ = build_lut(silu_fp32, s["conv_out"], s["u_post_conv_silu"])
        lut_silu_z, silu_z_scale = build_lut(silu_fp32, s["z_gate"], None)
```

- The softplus LUT maps the `s_dt_proj_out` domain to the `s_delta` domain.
- The SiLU-after-conv LUT maps the `s_conv_out` domain to the
  `s_u_post_conv_silu` domain.
- The SiLU-on-`z` LUT maps the `s_z_gate` domain to a new scale, `s_silu_z`. No
  independent calibration exists for this output. The function derives the scale
  from the function's own range over the input domain.

Each LUT uses 256 bytes of flash.

### The block, step by step

The excerpts come from `ssm_block_step_true_int8` in
`mcu/ssm_true_int8_src/ssm_backbone.c`.

**Step 1. `in_proj`, split, and requantize.** The two halves use separate
output scales.

```c
	ssm_qlinear_real(x_norm_q, w->in_proj_u_w_q, SSM_D_INNER, SSM_D_MODEL,
			w->s_norm_out, w->in_proj_w_scale, NULL, u_raw_real);
	ssm_qlinear_real(x_norm_q, w->in_proj_z_w_q, SSM_D_INNER, SSM_D_MODEL,
			w->s_norm_out, w->in_proj_w_scale, NULL, z_real);
	for (int c = 0; c < SSM_D_INNER; c++) {
		u_raw_q[c] = ssm_requantize(u_raw_real[c], w->s_conv_out);
		z_q[c] = ssm_requantize(z_real[c], w->s_z_gate);
	}
```

**Step 2. Depthwise causal convolution and SiLU LUT.** The convolution
accumulates in int32. All taps share the `s_conv_out` domain because
`conv_hist_q` stores only past `u_raw_q` values.

```c
	for (int c = 0; c < SSM_D_INNER; c++) {
		const int8_t *w_row = &w->conv_w_q[c * SSM_D_CONV];
		int32_t acc = 0;
		for (int k = 0; k < SSM_D_CONV - 1; k++) {
			acc += (int32_t) w_row[k] * (int32_t) conv_hist_q[c][k];
		}
		acc += (int32_t) w_row[SSM_D_CONV - 1] * (int32_t) u_raw_q[c];

		float conv_real = (float) acc * w->s_conv_out * w->conv_w_scale + w->conv_b[c];
		int8_t conv_out_q = ssm_requantize(conv_real, w->s_conv_out);
		u_q[c] = ssm_apply_lut(w->lut_silu_conv, conv_out_q);
		/* ... shift conv_hist_q, store u_raw_q[c] ... */
	}
```

**Step 3. `x_proj`, split into `delta_low`, `B`, and `C`.** The product is
computed once in real units. The code then requantizes each part to its own
scale.

```c
	ssm_qlinear_real(u_q, w->x_proj_w_q, SSM_DT_RANK + 2 * SSM_D_STATE, SSM_D_INNER,
			w->s_u_post_conv_silu, w->x_proj_w_scale, NULL, x_dbl_real);

	for (int i = 0; i < SSM_DT_RANK; i++) {
		delta_low_q[i] = ssm_requantize(x_dbl_real[i], w->s_x_proj_delta_low_out);
	}
	for (int n = 0; n < SSM_D_STATE; n++) {
		B_q[n] = ssm_requantize(x_dbl_real[SSM_DT_RANK + n], w->s_B);
		C_q[n] = ssm_requantize(x_dbl_real[SSM_DT_RANK + SSM_D_STATE + n], w->s_C);
	}
```

**Step 4. `dt_proj` with bias, then the softplus LUT.**

```c
	ssm_qlinear_real(delta_low_q, w->dt_proj_w_q, SSM_D_INNER, SSM_DT_RANK,
			w->s_x_proj_delta_low_out, w->dt_proj_w_scale, w->dt_proj_b, delta_raw_real);
	int8_t delta_q[SSM_D_INNER];
	for (int c = 0; c < SSM_D_INNER; c++) {
		int8_t delta_raw_q = ssm_requantize(delta_raw_real[c], w->s_dt_proj_out);
		delta_q[c] = ssm_apply_lut(w->lut_softplus, delta_raw_q);
	}
```

**Step 5. Discretization and the recurrence.** This step is the core of the
scheme. For each channel `c` and state index `n`, the code:

1. Computes `A_bar` in real units from `delta` and the fp32 `A`, applies the
   clamp at $-1.9$, and requantizes to `A_bar_q`.
2. Computes `B_bar` from the raw int8 product `delta_q * B_q` (an int32), then
   scales and requantizes to `B_bar_q`.
3. Computes the two state terms as int32 products, scales each to real units,
   and adds them.
4. Requantizes the sum to `ssm_h_t`, with the clip set by `SSM_H_CLIP`.
5. Accumulates `h * C_q` in an int32 for the scan output.

```c
	for (int c = 0; c < SSM_D_INNER; c++) {
		float delta_real = (float) delta_q[c] * w->s_delta;

		int32_t y_scan_acc = 0;
		for (int n = 0; n < SSM_D_STATE; n++) {
			float deltaA_real = delta_real * w->A[c * SSM_D_STATE + n];
			if (deltaA_real < -1.9f) {
				deltaA_real = -1.9f;
			}
			float A_bar_real = 1.0f + deltaA_real;
			int8_t A_bar_q = ssm_requantize(A_bar_real, w->s_A_bar);

			float B_bar_real = (float) ((int32_t) delta_q[c] * (int32_t) B_q[n])
					* w->s_delta * w->s_B;
			int8_t B_bar_q = ssm_requantize(B_bar_real, w->s_B_bar);

			float Ah_real = (float) ((int32_t) A_bar_q * (int32_t) h[c][n])
					* w->s_A_bar * w->s_h;
			float Bu_real = (float) ((int32_t) B_bar_q * (int32_t) u_q[c])
					* w->s_B_bar * w->s_u_post_conv_silu;
			float new_h_real = Ah_real + Bu_real;

			float new_h_q_f = roundf(new_h_real / w->s_h);
			if (new_h_q_f > (float) SSM_H_CLIP) {
				new_h_q_f = (float) SSM_H_CLIP;
			}
			if (new_h_q_f < -(float) SSM_H_CLIP) {
				new_h_q_f = -(float) SSM_H_CLIP;
			}
			ssm_h_t new_h_q = (ssm_h_t) new_h_q_f;
			h[c][n] = new_h_q;

			y_scan_acc += (int32_t) new_h_q * (int32_t) C_q[n];
		}
```

In equation form:

$$\bar{A}_q = Q_{s_{\bar A}}\left(1 + \max(\delta A, -1.9)\right), \qquad \bar{B}_q = Q_{s_{\bar B}}\left(s_\delta s_B \, \delta_q B_q\right)$$

$$h_{t,q} = Q_{s_h}\left(s_{\bar A} s_h \, \bar{A}_q h_{t-1,q} + s_{\bar B} s_u \, \bar{B}_q u_q\right)$$

Here $Q_s(\cdot)$ is the requantize function with scale $s$ and clip
$\pm$`SSM_H_CLIP`.

**Step 6. Scan output, D shortcut, gate, and `y_gated`.**

```c
		float y_scan_real = (float) y_scan_acc * w->s_h * w->s_C;
		int8_t y_scan_q = ssm_requantize(y_scan_real, w->s_y_scan);

		int8_t silu_z_q = ssm_apply_lut(w->lut_silu_z, z_q[c]);
		float y_pre_gate_real = (float) y_scan_q * w->s_y_scan
				+ w->D[c] * (float) u_q[c] * w->s_u_post_conv_silu;
		float y_gated_real = y_pre_gate_real * ((float) silu_z_q * w->s_silu_z);
		y_gated_q[c] = ssm_requantize(y_gated_real, w->s_y_gated);
	}
```

**Step 7. `out_proj`.**

```c
	ssm_qlinear_real(y_gated_q, w->out_proj_w_q, SSM_D_MODEL, SSM_D_INNER,
			w->s_y_gated, w->out_proj_w_scale, NULL, block_out_real);
	ssm_quantize_arr(block_out_real, w->s_block_output, SSM_D_MODEL, block_out_q);
```

### Residual stream, final norm, and pooling

The residual stream `x_real` stays in float. Each block adds its dequantized
output. RMSNorm runs in fp32 before each block and before the final norm. Its
output is quantized to int8 before the next integer operation. This excerpt is
`SSMBackbone_ProcessFrame`:

```c
	for (int layer = 0; layer < SSM_N_LAYERS; layer++) {
		const ssm_true_int8_layer_t *w = &ssm_layers[layer];

		float x_norm_real[SSM_D_MODEL];
		ssm_rmsnorm(x_real, w->norm_w, x_norm_real);
		int8_t x_norm_q[SSM_D_MODEL];
		ssm_quantize_arr(x_norm_real, w->s_norm_out, SSM_D_MODEL, x_norm_q);

		int8_t block_out_q[SSM_D_MODEL];
		ssm_block_step_true_int8(w, state->h[layer], state->conv_hist_q[layer],
				x_norm_q, block_out_q);

		for (int i = 0; i < SSM_D_MODEL; i++) {
			x_real[i] += (float) block_out_q[i] * w->s_block_output;
		}
	}
```

The pooling sum accumulates the int8 round trip of each frame in real units.
The pooled mean is then requantized to `ssm_final_norm_scale`, so the embedding
that leaves the backbone is an exact int8 round trip:

```c
void SSMBackbone_GetPooled(const SSMBackbone_State *state, float *pooled_out) {
	float inv_n = 1.0f / (float) state->frame_count;
	for (int i = 0; i < SSM_D_MODEL; i++) {
		float pooled_real = state->pooled_sum[i] * inv_n;
		int8_t pooled_q = ssm_requantize(pooled_real, ssm_final_norm_scale);
		pooled_out[i] = (float) pooled_q * ssm_final_norm_scale;
	}
}
```

This property is what makes the int8 head valid (see Scheme 4).

### What stays in float

| Item | Note |
| :--- | :--- |
| RMSNorm sum of squares and reciprocal square root | The source states that no clean int8 form exists. The output is still quantized. |
| Biases | The source states that the code adds them in real units at each rescale point. |
| `A`, `D`, `norm_w` | Static fp32 arrays, never requantized at run time. |
| Requantize arithmetic | Float multiply by the scales, float divide by the output scale, and `roundf`. |
| Residual stream `x_real` | Accumulates in float. |
| `A_bar` computation | The code multiplies `delta_real` by fp32 `A`, clamps, and then requantizes. |

### Overflow bounds

The `int32_t` accumulators are safe. For a dot product, the bound is
$127 \times 127 \times n$ for length $n$. For `SSM_D_INNER` = 128, this is
2,064,512, far below the int32 limit of about $2.1 \times 10^9$. For the scan
accumulator at `int16` width, the bound is
$32767 \times 127 \times$ `SSM_D_STATE`. For `SSM_D_STATE` = 8, this is
33,291,272.

## Scheme 4: heads

Each setup folder scores one head, either `euclidean` or `knn16`. The C struct
`SSMHeadResult` keeps all four score fields in every folder. The inactive head
writes a sentinel (score `-1.0`, anomaly `0`).

The exporter picks one default threshold, `percentile_same_machine_99.0`
(`DEFAULT_THRESHOLD_METHOD`). `diagnostics.json` holds all 14 methods.

### fp32 head

Two implementations exist in the template strings in
`export_deploy_matrix.py`. Both use the same distance function:

```c
static float ssm_l2_distance(const float *a, const float *b, int n) {
	float acc = 0.0f;
	for (int i = 0; i < n; i++) {
		float d = a[i] - b[i];
		acc += d * d;
	}
	return sqrtf(acc);
}
```

`euclidean` compares the embedding with one centroid (`ssm_ref_centroid`).
`knn16` takes the smallest distance to 16 cluster centers
(`ssm_ref_clusters`, fitted with k-means).

### int8 head

The int8 head is valid only on a true int8 backbone, because the pooled
embedding must be an exact int8 round trip at `ssm_final_norm_scale`. The
whole decision path runs in int32. Only the reported score uses one `sqrtf`.

The C code below is emitted into `ssm_distance_head.c`
(`true_int8_h16_euclidean_int8/ssm_distance_head.c`):

```c
static int8_t ssm_head_quantize(float real) {
	float q = roundf(real / ssm_final_norm_scale);
	if (q > 127.0f) {
		q = 127.0f;
	}
	if (q < -127.0f) {
		q = -127.0f;
	}
	return (int8_t) q;
}

static int32_t ssm_sumsq(const int8_t *a, const int8_t *b, int n) {
	int32_t acc = 0;
	for (int i = 0; i < n; i++) {
		int32_t d = (int32_t) a[i] - (int32_t) b[i];
		acc += d * d;
	}
	return acc;
}

void SSMDistanceHead_Score(const float *embedding, SSMHeadResult *result) {
	int8_t embedding_q[SSM_D_MODEL];
	for (int i = 0; i < SSM_D_MODEL; i++) {
		embedding_q[i] = ssm_head_quantize(embedding[i]);
	}

	result->euclidean_sumsq = ssm_sumsq(embedding_q, ssm_ref_centroid_q, SSM_D_MODEL);
	result->euclidean_score = sqrtf((float) result->euclidean_sumsq) * ssm_final_norm_scale;
	result->euclidean_anomaly = (result->euclidean_sumsq > ssm_threshold_euclidean_sumsq) ? 1 : 0;
	/* ... knn16 fields set to sentinel ... */
}
```

The decision compares an int32 sum of squares with an int32 threshold. No float
comparison happens.

The reference vectors and the threshold are quantized at export time against
the same scale as the embedding. The exporter never derives a separate scale.
From `quantize_ref_int8` and `emit_head_ref_and_module` in
`export_deploy_matrix.py`:

```python
def quantize_ref_int8(real_arr, scale):
    scaled = np.asarray(real_arr, dtype=np.float64) / scale
    n_clipped = int(np.sum(np.abs(scaled) > 127.0))
    if n_clipped > 0:
        print(f"      WARNING: {n_clipped} reference entr"
              f"{'y' if n_clipped == 1 else 'ies'} clip at int8 export "
              f"(magnitude exceeded the embedding's int8 range).")
    return np.clip(np.round(scaled), -127, 127).astype(np.int8)
```

```python
        thr_e_sumsq = int(round((chosen_threshold / final_norm_scale) ** 2)) if head == "euclidean" else -1
```

The threshold conversion in equation form:

$$T_{\text{sumsq}} = \operatorname{round}\left(\left(\frac{T}{s_{\text{final}}}\right)^2\right)$$

## Classic branch: the recurrence axis

The classic (`selective=False`) matrix has 30 setups. The selective matrix has
20. The classic backbone has no `x_proj` and no `dt_proj`. It stores `delta`,
`B`, and `C` as static parameters, so `A_bar` and `B_bar` are constants. This
creates one extra axis with two values:

| Form | Ships (fake quantization) | Ships (true int8) | Firmware work per frame | Flash |
| :--- | :--- | :--- | :--- | :--- |
| `qab` | int8 `A_log`, `dt`, `B`, `C` | int8 `delta_q` (`softplus(dt)` computed at export), int8 `B_q`, int8 `C_q`, fp32 `A` | Computes `A_bar` and `B_bar` from the shipped values. Fake quantization also runs `softplus` and `expf` on each access. | Lower |
| `qabar` | int8 `A_bar`, `B_bar`, `C`. `A_log`, `dt`, `B` do not ship. | int8 `A_bar_q`, `B_bar_q`, `C_q` | Reads the baked arrays. No discretize step. | Higher (two `d_inner x d_state` arrays) |

Both forms use the same `ranges.json` entries (`block<i>.A_bar` and
`block<i>.B_bar`). This keeps the `h`-width axis comparable across forms.

The classic `ssm_backbone.c` text is the same for both forms. Macros in the
generated header select the form.

Fake-quant classic, `qab` (`w_all_pertensor_qab_euclidean_fp32/ssm_weights.h`):

```c
#define SSM_DELTA(w, c) ssm_softplus(SSM_DT((w), (c)))
#define SSM_A_BAR(w, c, n, d) ssm_euler_abar((d) * SSM_A((w), (c), (n)))
#define SSM_B_BAR(w, c, n, d) ((d) * SSM_B((w), (n)))
```

Fake-quant classic, `qabar`
(`w_all_perchannel_qabar_euclidean_fp32/ssm_weights.h`). The macro discards
`delta` with the comma operator:

```c
#define SSM_DELTA(w, c) (0.0f)
#define SSM_A_BAR(w, c, n, d) ((void)(d), SSM_A_BAR_BAKED((w), (c), (n)))
#define SSM_B_BAR(w, c, n, d) ((void)(d), SSM_B_BAR_BAKED((w), (c), (n)))
```

True int8 classic, `qab` (`true_int8_h8_qab_euclidean_fp32/ssm_weights.h`).
The code recomputes and requantizes `A_bar` and `B_bar` on every frame:

```c
#define SSM_TI_DELTA_REAL(w, c) ((float) (w)->delta_q[(c)] * (w)->s_delta)
#define SSM_TI_A_BAR_Q(w, c, n, d) ssm_requantize(ssm_euler_abar((d) * (w)->A[(c) * SSM_D_STATE + (n)]), (w)->s_A_bar)
#define SSM_TI_B_BAR_Q(w, c, n, d) ssm_requantize((float) ((int32_t) (w)->delta_q[(c)] * (int32_t) (w)->B_q[(n)]) * (w)->s_delta * (w)->s_B, (w)->s_B_bar)
```

True int8 classic, `qabar` (`true_int8_h8_qabar_euclidean_fp32/ssm_weights.h`).
The code reads baked int8 values:

```c
#define SSM_TI_DELTA_REAL(w, c) (0.0f)
#define SSM_TI_A_BAR_Q(w, c, n, d) ((void) (d), (w)->A_bar_q[(c) * SSM_D_STATE + (n)])
#define SSM_TI_B_BAR_Q(w, c, n, d) ((void) (d), (w)->B_bar_q[(c) * SSM_D_STATE + (n)])
```

The classic recurrence loop then calls these macros:

```c
	for (int c = 0; c < SSM_D_INNER; c++) {
		const float delta_c = SSM_TI_DELTA_REAL(w, c);

		int32_t y_scan_acc = 0;
		for (int n = 0; n < SSM_D_STATE; n++) {
			int8_t A_bar_q = SSM_TI_A_BAR_Q(w, c, n, delta_c);
			int8_t B_bar_q = SSM_TI_B_BAR_Q(w, c, n, delta_c);
```

The classic true int8 scheme also drops the softplus LUT. `delta = softplus(dt)`
is a constant, so the exporter computes it once. The exporter handles `qabar`
in fake-quant mode through `ActivationQuantizer(group="discretized_const",
scale_source="onthefly")`, because no state-dict entry exists for `A_bar`. On
this branch the result is identical to weight quantization, because `A_bar` and
`B_bar` are constants.

## Setup identifiers

Table 4 maps the setup folder names to the schemes. `<head>` is `euclidean` or
`knn16`. Each identifier appears under
`mcu/deploy/case<N>/<model_hash>/setups/`.

| Scheme | Selective identifier | Classic identifier | Selective count | Classic count |
| :--- | :--- | :--- | ---: | ---: |
| Weight-only int8 | `w_<proj\|all>_<perchannel\|pertensor>_<head>_fp32` | `w_all_<perchannel\|pertensor>_<qab\|qabar>_<head>_fp32` | 8 | 8 |
| Weight + boundary activations | `w_all_pertensor_actboundaries_<head>_fp32` | `w_all_pertensor_actboundaries_<qab\|qabar>_<head>_fp32` | 2 | 4 |
| True int8 | `true_int8_<h8\|h16>_<head>_<fp32\|int8>` | `true_int8_<h8\|h16>_<qab\|qabar>_<head>_<fp32\|int8>` | 8 | 16 |
| Full fp32 | `full_fp32_<head>_fp32` | `full_fp32_<head>_fp32` | 2 | 2 |
| **Total** | | | **20** | **30** |

The classic full-fp32 baseline uses `qab` only. `qabar` always bakes and
quantizes `A_bar` and `B_bar`, so it cannot express an unquantized backbone.

## Open items

- **The classic setup matrix function may return early.** In the file text
  read for this finding, the blocks for combos 13 to 28, combos 29 to 30, and
  `return setups` in `build_classic_setup_matrix` appear indented under the
  `for recurrence in RECURRENCES:` loop of combos 9 to 12. If that indentation
  is real, the function returns after the first recurrence and yields 28
  setups, without the two `qabar` boundary combos. The deploy folders for
  `b77482e85dc3` contain all 30 setups. Either an earlier version of the file
  produced them, or the indentation read here is not what runs. The git
  history was not checked. To check, run
  `python -c "from mcu.export_deploy_matrix_classic import build_classic_setup_matrix as b; print(len(b()))"`.
  The expected output is `30`.
- **Selective source comments use old dimensions.** The comment above
  `ssm_qlinear_real` in `ssm_true_int8_src/ssm_backbone.c` uses 64 as the
  largest input dimension. The comment in the recurrence loop uses
  `SSM_D_STATE=16`. The generated header for `086acf0275b8` sets
  `SSM_D_INNER` to 128 and `SSM_D_STATE` to 8. The int32 bounds still hold. The
  classic source already states the 128 case.
- **Not read:** the last 366 lines of `export_deploy_matrix.py` (threshold
  computation, README renderer, `main`), lines 261 to 405 of
  `true_int8_sim_classic.py`, and the folder `mcu/ssm_head_src/`.

<!-- Claude comment: The two fake-quant schemes and the true int8 scheme answer
different questions. Fake quantization tests whether int8 rounding noise is
tolerable. It gives a flash saving and no compute saving, because every read
still does an int8-to-float conversion and a float multiply. True int8 tests
whether chained integer arithmetic survives. It is a hybrid, not a pure
integer pipeline: the requantize step multiplies by float scales, divides by
the output scale, and calls roundf. A production integer kernel would replace
that with an integer multiplier and shift. On a core without an FPU, for
example the RP2040's Cortex-M0+, each of those float operations runs in
software. Check this before you claim a latency benefit for true int8 on that
board. `findings/570` (not read for this draft) may already cover the
RP2040 case. -->

<!-- Claude comment: `u_raw_q` and `conv_out_q` share one scale, `s_conv_out`.
The calibration hook in `calibrate_missing_ranges` measures only the output of
the convolution module. No hook measures the range of `u_raw` itself. If the
range of `u_raw` is larger than the range of the convolution output, the
requantize step clips `u_raw`, and the clipped values also enter `conv_hist_q`.
I did not measure this. A five-minute check is to record `max|u_raw|` beside
`max|conv_out|` for the calibration clips and compare them. -->

<!-- Claude comment: Python rounds half to even (`np.round`), and C `roundf`
rounds half away from zero. The two differ only when `x / s` is exactly a
multiple of 0.5. That case is rare in float arithmetic, and the file name of
`findings/542` says the port is bit-exact, so the effect is probably small. I
did not test it, and I did not read `findings/542`.
It is a candidate explanation if a future parity check shows a one-LSB
difference at a single element. -->

<!-- Claude comment: In per-channel mode the exporter uses a scale of `1e-12`
for an all-zero row, and `quantize_dequantize` in `weight_quant_parity.py`
uses `1.0` for the same case. The quantized value is zero in both, so the
result does not change. The two constants differ only in the stored scale. -->
