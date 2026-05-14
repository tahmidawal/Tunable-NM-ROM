"""Tunable NM-ROM for the parametric Poisson equation (2D and 3D)."""

from .models.encoder import ViTEncoder
from .models.decoder import LinearCPDecoder
from .models.autoencoder import ViTLinearCPAutoencoder

__all__ = ["ViTEncoder", "LinearCPDecoder", "ViTLinearCPAutoencoder"]
__version__ = "0.1.0"
