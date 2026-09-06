import torch
from src.models.backbone import SSMBackbone

model = SSMBackbone(d_model=64, d_state=8, n_layers=2, expand=2, selective=True,
                    discretization='euler', pooling='mean').eval()
x = torch.randn(2, 344, 64)

with torch.no_grad():
    y_batched = model(x, mode='sequence', streaming=False)
    y_streaming = model(x, mode='sequence', streaming=True)

print((y_batched - y_streaming).abs().max().item())