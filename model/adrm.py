"""Adaptive Dual-Reconstruction Module (ADRM).

This is the canonical AMDRRec name for the historical autoencoder
implementation.  The inherited forward path is intentionally unchanged so
that the naming migration does not alter experiment semantics.
"""

from .autocoder import Autoencoder


class AdaptiveDualReconstructionModule(Autoencoder):
    """ADRM branch with attribute and edge reconstruction objectives."""


# Backward-compatible import name for old checkpoints and diagnostic scripts.
ADRM = AdaptiveDualReconstructionModule
