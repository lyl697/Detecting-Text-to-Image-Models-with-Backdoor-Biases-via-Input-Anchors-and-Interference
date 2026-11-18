"""Shared I/O, model loading, activation collection, and fixed image encoders."""
from __future__ import annotations
import csv
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from detection.feature_math import normalize
from detection.relations import SCHEMA
ROOT = Path(__file__).resolve().parents[1]

def local_path(path):
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("No prediction rows")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def resolve_model(value):
    path = local_path(value)
    if path.exists():
        return str(path.resolve())
    if Path(value).is_absolute() or value.startswith(("models/", "models\\")):
        raise FileNotFoundError(f"Set a real model/checkpoint path in defaults or arguments: {path}")
    return value  # Hugging Face repository id.


def source_identity(spec, args):
    result = {"revision": args.revision}
    for key in ("model_id", "lora_id"):
        value = spec.get(key) or ""
        if not value:
            result[key] = ""
            continue
        resolved = resolve_model(value)
        path = Path(resolved)
        info = {"source": resolved}
        if path.exists():
            files = [path] if path.is_file() else sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix in {".json", ".bin", ".safetensors", ".pt"})
            info["files"] = [(str(p.relative_to(path)) if path.is_dir() else p.name,
                              p.stat().st_size, p.stat().st_mtime_ns) for p in files]
        result[key] = info
    return result


def get_tensor(output, index=-1):
    import torch
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, dict):
        tensors = [v for v in output.values() if isinstance(v, torch.Tensor)]
    elif isinstance(output, (tuple, list)):
        tensors = [v for v in output if isinstance(v, torch.Tensor)]
    else:
        tensors = []
    return tensors[index] if tensors else None


def channel_vector(tensor, channel_axis=None):
    """Single prompt; select CFG conditional branch and pool non-channel axes."""
    if tensor.ndim not in (2, 3, 4) or tensor.shape[0] not in (1, 2):
        raise ValueError(f"Unsupported activation shape {tuple(tensor.shape)}; add a collection adapter")
    if channel_axis is None:
        channel_axis = 1 if tensor.ndim == 4 else -1  # Coarse BCHW or B-token-channel.
    axis = channel_axis % tensor.ndim
    if axis == 0:
        raise ValueError("Channel axis cannot be the batch axis")
    x = tensor[-1].detach().float()
    reduce_axes = tuple(i for i in range(x.ndim) if i != axis - 1)
    if reduce_axes:
        x = x.mean(dim=reduce_axes)
    return x.cpu().numpy()

class Collector:
    """Discover coarse blocks and supported leaf modules."""
    def __init__(self, root, config):
        import torch.nn as nn
        self.config, self.step, self.handles = config, 0, []
        self.blocks, self.modules = {}, {}
        block_items = []
        for name, child in root.named_children():
            if isinstance(child, nn.ModuleList):
                block_items.extend((f"{name}.{i}", block) for i, block in enumerate(child))
        # U-Net's middle block is not in a ModuleList.
        middle = getattr(root, "mid_block", None)
        if middle is not None and all(middle is not block for _, block in block_items):
            block_items.append(("mid_block", middle))
        if not block_items:
            raise ValueError("No coarse blocks found; implement a block collection adapter for this pipeline")
        self.block_names = [name for name, _ in block_items]
        for name, block in block_items:
            self.handles.append(block.register_forward_hook(self.make_hook(self.blocks, name, "block", True)))
        self.module_names = []
        wrapped_prefixes = []
        for name, module in root.named_modules():
            if not name or any(name.startswith(prefix + ".") for prefix in wrapped_prefixes):
                continue
            base_layer = getattr(module, "base_layer", None)
            observed_type = base_layer if isinstance(base_layer, nn.Module) else module
            if base_layer is None and any(True for _ in module.children()):
                continue
            if isinstance(observed_type, nn.Linear):
                kind = "linear"
            elif isinstance(observed_type, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                kind = "conv"
            elif isinstance(observed_type, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.RMSNorm)):
                kind = "norm"
            else:
                continue
            if isinstance(base_layer, nn.Module):
                wrapped_prefixes.append(name)
            self.module_names.append((name, kind))
            channel_axis = 1 if isinstance(observed_type, (
                nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)) else -1
            self.handles.append(module.register_forward_hook(
                self.make_hook(self.modules, name, kind, False, channel_axis)))
        if not self.module_names:
            self.close()
            raise ValueError("No supported linear/conv/norm leaf modules")

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()



def load_pipeline(spec, args):
    import torch
    from diffusers import DiffusionPipeline
    dtype = getattr(torch, args.dtype) if args.device != "cpu" else torch.float32
    source = resolve_model(spec["model_id"])
    options = {}
    index = Path(source) / 'model_index.json'
    if index.is_file():
        config = json.loads(index.read_text(encoding='utf-8'))
        if 'safety_checker' in config:
            options.update(safety_checker=None, requires_safety_checker=False)
    pipe = DiffusionPipeline.from_pretrained(source, torch_dtype=dtype,
                                            revision=args.revision, local_files_only=args.local_files_only, **options)
    if spec.get("lora_id"):
        pipe.load_lora_weights(resolve_model(spec["lora_id"]))
    pipe.to(args.device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if "callback_on_step_end" not in inspect.signature(pipe.__call__).parameters:
        raise ValueError("Pipeline must expose callback_on_step_end; add an adapter for other APIs")
    return pipe


def state_hash(model):
    """Hash the actual encoder weights, not merely its architecture/tag."""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        array = value.detach().cpu().contiguous().numpy()
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()


class ImageScorer:
    def __init__(self, args, metrics):
        from detection.clip_encoder import load_clip_encoder
        from detection.image_metrics import get_lpips_fn, get_ssim_fn
        self.backend, self.model, self.preprocess, encoder_identity = load_clip_encoder(args, args.device, ROOT)
        self.lpips = get_lpips_fn(args.device).requires_grad_(False).eval() if "lpips" in metrics else None
        self.ssim = get_ssim_fn(args.device).eval() if "ssim" in metrics else None
        self.device, self.metrics = args.device, metrics
        self.identity = {**encoder_identity, "weights_sha256": state_hash(self.model),
                         "lpips_sha256": state_hash(self.lpips) if self.lpips else None}
        self._embedding_cache = {}

    def to(self, device):
        for model in (self.model, self.lpips, self.ssim):
            if model is not None:
                model.to(device)
        self.device = device
        return self

    def embedding(self, path, image):
        import torch
        key = str(path)
        if key not in self._embedding_cache:
            with torch.inference_mode():
                inputs = self.preprocess(images=image, return_tensors="pt")
                feat = self.model(pixel_values=inputs["pixel_values"].to(self.device)).image_embeds
                feat = torch.nn.functional.normalize(feat.float(), dim=-1)
            # Keep embeddings on CPU, including reusable references.
            self._embedding_cache[key] = feat.cpu()
        return self._embedding_cache[key]

    def compare(self, target, reference):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        from detection.image_metrics import tensorize_for_lpips
        with Image.open(target) as im:
            a = im.convert("RGB")
        with Image.open(reference) as im:
            b = im.convert("RGB")
        if a.size != b.size:
            raise ValueError("Images must have a shared output resolution")
        output = {}
        with torch.inference_mode():
            if "clip_sim" in self.metrics:
                output["clip_sim"] = float((self.embedding(target, a) * self.embedding(reference, b)).sum())
            if self.lpips is not None:
                output["lpips"] = float(self.lpips(tensorize_for_lpips(a, self.device),
                                                   tensorize_for_lpips(b, self.device)).item())
            if "mse" in self.metrics or self.ssim is not None:
                at, bt = to_tensor(a).unsqueeze(0).to(self.device), to_tensor(b).unsqueeze(0).to(self.device)
                if "mse" in self.metrics:
                    output["mse"] = float(torch.mean((at - bt) ** 2).item())
                if self.ssim is not None:
                    output["ssim"] = float(self.ssim(at, bt).item())
                    self.ssim.reset()
        if any(m.startswith("hog_") for m in self.metrics):
            from skimage.color import rgb2gray
            from skimage.feature import hog
            from detection.image_metrics import _pearson_corr, _dtw_distance
            def histogram(im):
                vector = hog(rgb2gray(np.array(im.resize((256, 256)))),
                             orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                # Shared fixed bins for both descriptor-value histograms.
                counts, _ = np.histogram(vector, bins=np.linspace(0, 1, 65))
                return counts / max(counts.sum(), 1)
            ha, hb = histogram(a), histogram(b)
            output["hog_pearson"], output["hog_dtw"] = _pearson_corr(ha, hb), _dtw_distance(ha, hb)
        values = [output[m] for m in self.metrics]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite image metric; inspect images and scorer")
        return values


def read_features(path, schema=SCHEMA):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Feature dataset is empty")
    first = rows[0]
    keys = set()
    identities, model_sources = {}, {}
    for row in rows:
        if row.get("schema") != schema or row["signature"] != first["signature"] or row["feature_names"] != first["feature_names"]:
            raise ValueError("Mixed/incompatible feature schema or extraction protocol")
        key = row["model_id"], row["prompt_index"]
        if key in keys:
            raise ValueError("Duplicate model/prompt row")
        keys.add(key)
        source = row["source_identity"]
        if source in identities and identities[source] != row["model_id"]:
            raise ValueError("One checkpoint was assigned multiple model names")
        if row["model_id"] in model_sources and model_sources[row["model_id"]] != source:
            raise ValueError("One model name has conflicting checkpoint identities")
        model_sources[row["model_id"]] = source
        identities[source] = row["model_id"]
    x = np.asarray([r["values"] for r in rows], dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != len(first["feature_names"]) or not np.isfinite(x).all():
        raise ValueError("Invalid feature tensor")
    return rows, x
