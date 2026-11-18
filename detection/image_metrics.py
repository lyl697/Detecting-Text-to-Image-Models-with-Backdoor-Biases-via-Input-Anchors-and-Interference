"""VAE decoding and fixed image-comparison helpers."""
from __future__ import annotations
import math
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def _pipeline_device(pipe):
    return getattr(pipe, "device", getattr(pipe, "_execution_device", "cpu"))


def _decode_latents_to_image(pipe, latents: torch.Tensor, height: int, width: int) -> Image.Image:
    latents = latents.detach().to(_pipeline_device(pipe))

    # FLUX stores latents in a packed sequence layout. Diffusers exposes an
    # unpack helper on FluxPipeline in recent versions.
    if latents.dim() == 3 and hasattr(pipe, "_unpack_latents"):
        latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)

    latents = latents[:1].to(dtype=getattr(pipe.vae, "dtype", latents.dtype))
    scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
    shift_factor = getattr(pipe.vae.config, "shift_factor", None)
    if shift_factor is not None:
        latents = (latents / scaling_factor) + shift_factor
    else:
        latents = latents / scaling_factor

    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def get_lpips_fn(device: str):
    import lpips

    return lpips.LPIPS(net="alex").to(device).eval()


def get_ssim_fn(device: str):
    from torchmetrics import StructuralSimilarityIndexMeasure

    return StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


def tensorize_for_lpips(image: Image.Image, device: str):
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return to_tensor(image).unsqueeze(0).to(device)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a = a[:n]
    b = b[:n]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    dp = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i, ai in enumerate(a, start=1):
        for j, bj in enumerate(b, start=1):
            dp[i, j] = abs(ai - bj) + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[len(a), len(b)] / (len(a) + len(b)))
