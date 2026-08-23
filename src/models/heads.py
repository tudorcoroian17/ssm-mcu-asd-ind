import torch.nn as nn


class PredictionHead(nn.Module):
    """
    One linear layer, d_model -> n_mels. Attaches to the SEQUENCE
    output only — pooling collapses time, so there's no 'next frame'
    left once you pool. Dropped at inference for the distance-only
    config, retained for the fused S_recon variant (01_design_decisions.md §4).
    """
    def __init__(self, d_model, n_mels):
        super().__init__()
        self.proj = nn.Linear(d_model, n_mels)

    def forward(self, seq_output): # (batch, T, d_model) -> (batch, T, n_mels)
        return self.proj(seq_output)
