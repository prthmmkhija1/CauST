"""
caust/models/autoencoder.py
============================
Graph Attention Autoencoder for spatial transcriptomics.

Architecture (STAGATE-inspired, CauST re-implementation)
---------------------------------------------------------
Encoder  :  GAT1(in_dim → hidden_dim)  →  BN + ELU  →  GAT2(hidden_dim → latent_dim)
Decoder  :  Linear(latent_dim → in_dim)

The latent representations (30-D by default) are then clustered (KMeans)
to produce spatial domain assignments.

Why Graph Attention?
--------------------
A plain autoencoder treats each spot independently. A GAT autoencoder
lets each spot "attend" to its spatial neighbours — spots that are
physically close share information, which is exactly how tissue regions
work biologically (a cell is influenced by its immediate environment).
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class GATEncoder(nn.Module):
    """
    Two-layer Graph Attention Network encoder.

    Layer 1 : GATConv(in_dim, hidden_dim) + BatchNorm + ELU
    Layer 2 : GATConv(hidden_dim, latent_dim)

    Parameters
    ----------
    in_dim      : number of input genes
    hidden_dim  : intermediate hidden dimension (default 512)
    latent_dim  : size of the latent / embedding space (default 30)
    heads       : number of attention heads in layer 1
    dropout     : dropout rate on attention coefficients
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 30,
        heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,   # average multi-head outputs
            dropout=dropout,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=latent_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        z = self.conv2(x, edge_index)
        return z


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class LinearDecoder(nn.Module):
    """
    Simple linear decoder: latent_dim → in_dim.

    A minimal decoder forces the encoder to capture the most relevant
    biological variation in a compact latent space.
    """

    def __init__(self, latent_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


# ---------------------------------------------------------------------------
# Full autoencoder
# ---------------------------------------------------------------------------

class SpatialAutoencoder(nn.Module):
    """
    Graph Attention Autoencoder for spatial domain identification.

    Usage
    -----
        model = SpatialAutoencoder(in_dim=3000)
        model, losses = train_autoencoder(model, pyg_data, epochs=500)
        Z = model.get_latent(x_tensor, edge_index)   # (n_spots × 30)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 30,
        heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = GATEncoder(in_dim, hidden_dim, latent_dim, heads, dropout)
        self.decoder = LinearDecoder(latent_dim, in_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, edge_index)
        x_recon = self.decode(z)
        return z, x_recon

    @torch.no_grad()
    def get_latent(self, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
        """
        Inference-only: return latent embeddings as a NumPy array.
        No gradient computation (faster, uses less memory).
        """
        self.eval()
        z = self.encode(x, edge_index)
        return z.cpu().numpy()


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_autoencoder(
    model: SpatialAutoencoder,
    data: Data,
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[SpatialAutoencoder, List[float]]:
    """
    Train the SpatialAutoencoder with MSE reconstruction loss.

    The model learns to compress each spot's gene expression (along with
    its spatial neighbourhood context) into a 30-D latent vector, then
    reconstruct it.  The latent vectors encode the biologically meaningful
    variation we later use for clustering.

    Parameters
    ----------
    model        : SpatialAutoencoder instance
    data         : torch_geometric Data  (.x, .edge_index)
    epochs       : training epochs
    lr           : Adam learning rate
    weight_decay : L2 regularisation strength
    device       : 'cpu' or 'cuda'
    verbose      : print loss every 50 epochs

    Returns
    -------
    (trained_model, loss_history)
    """
    model = model.to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # Reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=50, factor=0.5, min_lr=1e-5
    )

    loss_history: List[float] = []
    model.train()

    epoch_iter = range(1, epochs + 1)
    if verbose:
        from tqdm import tqdm
        epoch_iter = tqdm(epoch_iter, desc="  Training", unit="ep",
                          bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for epoch in epoch_iter:
        optimizer.zero_grad()
        _, x_recon = model(data.x, data.edge_index)
        loss = F.mse_loss(x_recon, data.x)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        loss_val = loss.item()
        loss_history.append(loss_val)

        if verbose and hasattr(epoch_iter, 'set_postfix'):
            epoch_iter.set_postfix(loss=f"{loss_val:.6f}",
                                   lr=f"{optimizer.param_groups[0]['lr']:.1e}")

    model.eval()
    print(f"  Training complete. Final loss: {loss_history[-1]:.6f}")
    return model, loss_history
