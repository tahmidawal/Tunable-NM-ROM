"""Official NeuralOperator FNO with explicit double precision and field contract."""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import os
from pathlib import Path

import torch
from neuralop.models import FNO
from neuralop.layers.spectral_convolution import SpectralConv
from spectral_conv_f64 import SpectralConvF64, UPSTREAM_SHA256


def configure():
    if os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") != "highest":
        raise RuntimeError("JAX_DEFAULT_MATMUL_PRECISION=highest is mandatory")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA backend is mandatory")
    torch.set_default_dtype(torch.float64)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("torch_backend=cuda", torch.cuda.get_device_name(), flush=True)
    actual = hashlib.sha256(Path(inspect.getfile(SpectralConv)).read_bytes()).hexdigest()
    if actual != UPSTREAM_SHA256:
        raise RuntimeError(f"SpectralConv source differs from reviewed implementation: {actual}")
    return {"torch": torch.__version__, "neuraloperator": importlib.metadata.version("neuraloperator"),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "spectral_module_sha256": actual, "precision": "float64/complex128"}


def to_f64(model):
    # Module.double() leaves complex parameters at complex64. Preserve the
    # imaginary component while explicitly converting both parameter types.
    return model._apply(lambda x: x.to(dtype=torch.complex128 if x.is_complex() else torch.float64)
                        if x.is_floating_point() or x.is_complex() else x)


def make_model(pde, config):
    # Coordinates are constructed explicitly in float64, including both edges.
    model = FNO(n_modes=(config["modes"],) * 2,
                in_channels=3 if pde == "poisson" else 4,
                out_channels=1 if pde == "poisson" else 5,
                hidden_channels=config["width"], n_layers=config["layers"],
                positional_embedding=None, fno_block_precision="full",
                factorization=config.get("factorization"), rank=config.get("rank", 1.0),
                conv_module=SpectralConvF64)
    return to_f64(model).cuda()


def features(field, parameters, mean, std):
    batch, _, height, width = field.shape
    xx = torch.linspace(0, 1, height, device=field.device, dtype=torch.float64)
    yy = torch.linspace(0, 1, width, device=field.device, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(xx, yy, indexing="ij"))[None].expand(batch, -1, -1, -1)
    raw = torch.cat((field, parameters[:, :, None, None].expand(-1, -1, height, width)), dim=1)
    return torch.cat(((raw - mean) / std, grid), dim=1)


def predict(model, field, parameters, mean, std, output_scale, pde):
    output = model(features(field, parameters, mean, std)) * output_scale
    mask = torch.ones_like(output[:, :1])
    mask[..., 0, :] = 0
    mask[..., -1, :] = 0
    mask[..., :, 0] = 0
    mask[..., :, -1] = 0
    output = (output * mask)[:, :, None]
    return output if pde == "poisson" else torch.cat((field[:, None], output), dim=1)


def relative_errors(prediction, target, field, pde):
    numerator = (prediction - target).square().sum(dim=(-3, -2, -1)).sqrt()
    denom = (target[:, 0] if pde == "poisson" else field).square().sum(dim=(-3, -2, -1)).sqrt()
    return numerator / denom[:, None].clamp_min(1e-300)


def check_dtypes(model, gradients=False):
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.is_floating_point() or value.is_complex():
            expected = torch.complex128 if value.is_complex() else torch.float64
            if value.dtype != expected:
                raise RuntimeError(f"Parameter/buffer precision failure: {name} {value.dtype}")
            if gradients and value.grad is not None and value.grad.dtype != expected:
                raise RuntimeError(f"Gradient precision failure: {name}")
