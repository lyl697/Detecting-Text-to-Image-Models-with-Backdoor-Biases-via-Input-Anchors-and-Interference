"""Only the image measurements used by the Stage-II formula."""
import numpy as np
from PIL import Image, ImageOps
from detection import common


def full_ssim(path_a, path_b, size=256):
    def pixels(path):
        with Image.open(path) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
            return np.asarray(rgb, dtype=float) / 255 @ np.array([.299, .587, .114])
    return ssim(pixels(path_a), pixels(path_b))


def background_ssim(path_a, mask_a, path_b, mask_b, size=256, min_common=.1):
    if mask_a is None or mask_b is None:
        return None, 0., 0
    a, ma = background_pixels(path_a, mask_a, size)
    b, mb = background_pixels(path_b, mask_b, size)
    common_mask = ma & mb
    coverage = float(common_mask.mean())
    if coverage < min_common or not common_mask.any():
        return None, coverage, 0
    windows = np.lib.stride_tricks.sliding_window_view(common_mask, (11, 11)).all(axis=(-2, -1))
    count = int(windows.sum())
    if not count:
        return None, coverage, 0
    weights = np.array([.299, .587, .114]) / 255
    a = np.where(common_mask[..., None], a, 128).astype(np.uint8)
    b = np.where(common_mask[..., None], b, 128).astype(np.uint8)
    return float(np.clip(ssim_map(a @ weights, b @ weights)[windows].mean(), -1, 1)), coverage, count


def sign_hash(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('VAE tensors must have matching finite spatial coordinates')
    return float(np.mean((a >= 0) == (b >= 0)))


class ImageFeatures:
    """Shared VAE and YOLO segmentation; no DINO/CLIP analysis-only dependencies."""
    def __init__(self, args):
        import torch
        from diffusers import AutoencoderKL
        from ultralytics import YOLO
        self.torch, self.args = torch, args
        vae_path = common.local_path(args.vae_model)
        if (vae_path / 'vae').is_dir():
            vae_path = vae_path / 'vae'
        self.vae = AutoencoderKL.from_pretrained(str(vae_path), local_files_only=True,
            torch_dtype=torch.float32).eval().requires_grad_(False).to(args.device)
        self.segmenter = YOLO(str(common.local_path(args.segmenter_weights)))
        if self.segmenter.task != 'segment':
            raise ValueError('Stage II requires YOLO segmentation weights')
        names = self.segmenter.names
        names = dict(enumerate(names)) if isinstance(names, list) else names
        self.classes = [int(k) for k, v in names.items() if v == 'person']
        if len(self.classes) != 1:
            raise ValueError('The segmenter must contain the person class')

    def encode(self, path):
        t, a = self.torch, self.args
        with Image.open(path) as source:
            im = ImageOps.exif_transpose(source).convert('RGB')
            pixels = np.asarray(im.resize((a.vae_size, a.vae_size), Image.Resampling.BICUBIC), dtype=np.float32)/255
            batch = t.from_numpy((pixels*2-1).transpose(2,0,1)[None]).to(a.device)
            with t.inference_mode():
                latent = self.vae.encode(batch).latent_dist.mode()[0].float().cpu().numpy()
            result = self.segmenter.predict(source=im, imgsz=a.segment_size, classes=self.classes,
                conf=.25, iou=.7, max_det=300, retina_masks=True, half=False,
                device=a.device, verbose=False)[0]
        boxes = result.boxes
        selected = np.isin(boxes.cls.detach().cpu().numpy(), self.classes) if boxes is not None else np.zeros(0, bool)
        xyxy = boxes.xyxy.detach().cpu().numpy()[selected] if boxes is not None else np.empty((0,4))
        if len(xyxy) and result.masks is None:
            raise ValueError('Instance boxes were detected but segmentation masks are missing')
        masks = result.masks.data.detach().cpu().numpy()[selected] if len(xyxy) else None
        p = instance_union(xyxy, masks, result.orig_shape, a.mask_size)
        regions, accepted, _ = mask_regions(p, .5, a.mask_margin, .05)
        # Match the recorded experiment: zero detections uses the full image as background.
        fallback = len(xyxy) == 0
        background = np.ones_like(p, dtype=bool) if fallback else regions['background'] if accepted['background'] else None
        return dict(latent=latent, background=np.array([]) if background is None else background,
                    full_image_background=np.array(fallback))

def gaussian_mean(x, window=11, sigma=1.5):
    """Valid Gaussian convolution, excludes boundary pixels."""
    radius = window//2
    weights = np.exp(-np.arange(-radius, radius+1, dtype=float)**2/(2*sigma**2))
    weights /= weights.sum()
    out = np.einsum('ijk,k->ij', np.lib.stride_tricks.sliding_window_view(x, window, axis=1), weights)
    return np.einsum('ijk,k->ij', np.lib.stride_tricks.sliding_window_view(out, window, axis=0), weights)


def ssim_map(a, b, window=11, sigma=1.5):
    """Local grayscale SSIM map, population covariance, K1=.01/K2=.03, range=1."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (a.shape != b.shape or a.ndim != 2 or window < 3 or window % 2 != 1
            or min(a.shape) < window or not np.isfinite(sigma) or sigma <= 0):
        raise ValueError('SSIM needs equally sized 2D images at least window pixels wide; odd window >=3')
    ma, mb = gaussian_mean(a, window, sigma), gaussian_mean(b, window, sigma)
    va = np.maximum(0, gaussian_mean(a*a, window, sigma)-ma*ma)
    vb = np.maximum(0, gaussian_mean(b*b, window, sigma)-mb*mb)
    cov = gaussian_mean(a*b, window, sigma)-ma*mb
    score = ((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return score


def ssim(a, b):
    return float(np.clip(ssim_map(a, b).mean(), -1, 1))


def instance_union(boxes, masks, original_shape, output_size):
    """Union selected instances in original image coordinates, then square-resize.

    Masks must be unpadded original-resolution masks (Ultralytics retina_masks).
    The empty result stays empty; the caller records the zero-detection fallback.
    """
    from PIL import Image
    height, width = map(int, original_shape)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if height <= 0 or width <= 0 or not np.isfinite(boxes).all():
        raise ValueError('Invalid instance boxes or original image shape')
    foreground = np.zeros((height, width), dtype=np.uint8)
    if len(boxes):
        masks = np.asarray(masks)
        if masks.shape != (len(boxes), height, width) or not np.isfinite(masks).all():
            raise ValueError('YOLO masks must match instance count and original image size; enable retina_masks')
        foreground = np.any(masks >= .5, axis=0).astype(np.uint8)
    return np.asarray(Image.fromarray(foreground).resize((output_size, output_size),
                                                       Image.Resampling.NEAREST), dtype=np.float32)


def mask_regions(probability, threshold=.5, margin=8, min_area=.05):
    """Binary foreground, and background outside a dilated foreground boundary."""
    from PIL import Image, ImageFilter
    p = np.asarray(probability, dtype=np.float32)
    if p.ndim != 2 or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("Expected a finite H x W probability mask in [0,1]")
    foreground = p >= threshold
    image = Image.fromarray(foreground.astype(np.uint8) * 255)
    dilated = np.asarray(image.filter(ImageFilter.MaxFilter(2 * margin + 1))) > 0 if margin else foreground
    background = ~dilated
    masks = dict(foreground=foreground, background=background)
    valid = {k: min_area <= float(v.mean()) <= 1-min_area for k, v in masks.items()}
    quality = {k + "_fraction": float(v.mean()) for k, v in masks.items()}
    quality['uncertain_fraction'] = float(((p > .1) & (p < .9)).mean())
    quality['excluded_boundary_fraction'] = float((dilated & ~foreground).mean())
    return masks, valid, quality


def background_pixels(path, mask, size):
    """Masked interpolation prevents foreground RGB from leaking into background."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.size:
        raise ValueError('Background mask must be a nonempty 2D mask')
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert('RGB')
        with Image.fromarray(mask.astype(np.uint8)) as image:
            native_mask = np.asarray(image.resize(rgb.size, Image.Resampling.NEAREST), dtype=np.float32)
        values = np.asarray(rgb, dtype=np.float32)
    def resize(array):
        with Image.fromarray(array.astype(np.float32)) as im:
            return np.asarray(im.resize((size, size), Image.Resampling.BILINEAR), dtype=float)
    alpha = resize(native_mask)
    filtered = np.stack([resize(values[..., c]*native_mask) for c in range(3)], axis=-1)
    filtered /= np.maximum(alpha[..., None], 1e-12)
    return np.clip(filtered, 0, 255).round().astype(np.uint8), alpha >= 1-1e-6
