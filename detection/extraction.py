"""Stage-I feature extraction: block relations, leaf relations, latent dynamics, and image metrics."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from detection import common
from detection.feature_math import TRACE_NAMES, latent_trajectory
from detection.relations import SCHEMA, RAW_SCHEMA, group_relations, lfd_relation_features
from detection.settings import protocol, model_specs, native_capture_grid, feature_layout

def load_anchors(args):
    if args.anchor_prompts:
        anchors = [s.strip() for s in common.local_path(args.anchor_prompts).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    else:
        anchors = common.local_path("data/anchors.txt").read_text(encoding="utf-8").splitlines()
    if args.max_anchors < 0:
        raise ValueError("max_anchors cannot be negative")
    if args.max_anchors:
        anchors = anchors[:args.max_anchors]
    if len(anchors) < 2 or len(set(anchors)) != len(anchors):
        raise ValueError("Need at least two unique, ordered anchors")
    return anchors


def source_identity(spec, args):
    for key in ("model_id", "lora_id"):
        value = spec.get(key) or ""
        if value.startswith((".", "~")) and not common.local_path(value).exists():
            raise FileNotFoundError(f"Set the real local {key} path: {common.local_path(value)}")
    return common.source_identity(spec, args)


def sample_directory(directory, prompt, seed):
    return directory / "samples" / common.digest({"prompt": prompt, "seed": seed})


def relation_file(directory, prompt):
    return directory / (common.digest(prompt) + ".npz")


def write_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


class RelationCollector(common.Collector):
    """Reuse module discovery/LoRA handling; retain raw channel coordinates."""
    def make_hook(self, destination, name, kind, required, channel_axis=None):
        def hook(module, inputs, output):
            if self.step not in self.config["steps"]:
                return
            tensor = common.get_tensor(output, self.config["block_output_index"])
            if tensor is None or tensor.ndim not in (2, 3, 4) or tensor.shape[0] not in (1, 2):
                if required:
                    raise ValueError(f"Cannot interpret block {name}; add a collection adapter")
                return
            vector = np.array(common.channel_vector(tensor, channel_axis), dtype=np.float32, copy=True)
            if not vector.size or not np.isfinite(vector).all():
                raise ValueError(f"Invalid activation vector in {name}")
            destination.setdefault(self.step, {}).setdefault((name, kind), []).append(vector)
        return hook

    def snapshot(self, step):
        blocks = self.blocks.pop(step, {})
        modules = self.modules.pop(step, {})
        if len(blocks) != len(self.block_names):
            raise ValueError(f"Missing coarse blocks at step {step}")
        result = {}
        for collection in (blocks, modules):
            for (name, kind), values in collection.items():
                if len({v.shape for v in values}) != 1:
                    raise ValueError(f"Module {name} emitted inconsistent channel widths in one step")
                result[(kind, name)] = np.mean(np.stack(values), axis=0)
        return result


def capture_one(pipe, prompt, seed, args, config, directory):
    import torch
    from detection.image_metrics import _decode_latents_to_image
    capture = config["capture"]
    if args.denoiser_component == "auto":
        root = getattr(pipe, "unet", None)
        if root is None:
            root = getattr(pipe, "transformer", None)
    else:
        root = getattr(pipe, args.denoiser_component, None)
    if root is None:
        raise ValueError("No denoiser found; set denoiser_component")
    # Do not mutate the shared extraction protocol when remapping one pipeline.
    collector = RelationCollector(root, dict(capture))
    snapshots, latents, observed_timesteps = {}, {}, {}
    expected_count = capture["num_inference_steps"]
    native_to_slot = dict(zip(capture["steps"], capture["steps"]))
    directory.mkdir(parents=True, exist_ok=True)

    def callback(callback_pipe, step, timestep, kwargs):
        nonlocal expected_count, native_to_slot
        if step != collector.step:
            raise ValueError(f"Unexpected native callback order: got {step}, expected {collector.step}")
        if step == 0:
            # Pipelines initialize their scheduler inside __call__. The first hook
            # already captured step 0, which belongs to every supported grid.
            expected_count, native_to_slot = native_capture_grid(callback_pipe, capture)
            collector.config["steps"] = list(native_to_slot)
        if step in native_to_slot:
            slot = native_to_slot[step]
            snapshots[slot] = collector.snapshot(step)
            latents[slot] = kwargs["latents"].detach().float().cpu().numpy().copy()
            observed_timesteps[slot] = float(timestep)
            image = _decode_latents_to_image(callback_pipe, kwargs["latents"], args.height, args.width)
            # Slot labels remain stable for downstream fusion; metadata records
            # the actual native step and scheduler timestep for every slot.
            image.save(directory / f"step_{slot:03d}.png")
        collector.step = step + 1
        return kwargs

    try:
        with torch.inference_mode():
            pipe(prompt=prompt, generator=torch.Generator(device=args.device).manual_seed(seed),
                 num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                 height=args.height, width=args.width, output_type="latent",
                 callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
    finally:
        collector.close()
    if set(snapshots) != set(capture["steps"]) or collector.step != expected_count:
        scheduler_name = type(getattr(pipe, "scheduler", None)).__name__
        raise ValueError(
            f"Pipeline callback grid differs: scheduler={scheduler_name}, "
            f"requested={args.num_inference_steps}, expected_callbacks={expected_count}, "
            f"observed_callbacks={collector.step}, captured_slots={sorted(snapshots)}. "
            "Feature extraction uses fixed callback indices; inspect the pipeline callback contract.")
    names = sorted(set().union(*(d.keys() for d in snapshots.values())))
    arrays = {"names": np.asarray(names, dtype=str),
              "trace": latent_trajectory([latents[s] for s in capture["steps"]]),
              "native_steps": np.asarray(list(native_to_slot), dtype=np.int64),
              "scheduler_timesteps": np.asarray([observed_timesteps[s] for s in capture["steps"]]),
              "native_step_count": np.asarray(expected_count, dtype=np.int64)}
    for i, key in enumerate(names):
        entries = [snapshots[s].get(key) for s in capture["steps"]]
        shapes = {x.shape for x in entries if x is not None}
        if len(shapes) != 1:
            raise ValueError(f"Channel width changed over time within module {key}")
        shape = next(iter(shapes))
        arrays[f"v{i}"] = np.stack([x if x is not None else np.zeros(shape, np.float32) for x in entries])
        arrays[f"m{i}"] = np.array([x is not None for x in entries], dtype=bool)
    return arrays


def raw_complete(directory, capture):
    marker = directory / "complete.json"
    if not marker.is_file() or not (directory / "vectors.npz").is_file():
        return False
    try:
        if json.loads(marker.read_text(encoding="utf-8"))["schema"] != RAW_SCHEMA:
            return False
    except (ValueError, KeyError):
        return False
    return all((directory / f"step_{s:03d}.png").is_file() for s in capture["steps"])


def collect_model(spec, prompts, anchors, args, config):
    import torch
    identity = source_identity(spec, args)
    cache_key = common.digest({"source": identity, "capture": config["capture"]})
    directory = common.local_path(args.cache_dir) / cache_key
    all_prompts = list(dict.fromkeys(anchors + prompts))
    pending = [(prompt, r) for prompt in all_prompts for r in range(args.num_images)
               if not args.resume or not raw_complete(sample_directory(directory, prompt, args.seed + r), config["capture"])]
    if not pending:
        print(f"[cache] {spec['name']}: all {len(all_prompts)} prompts and anchors reused", flush=True)
        return directory
    pipe = common.load_pipeline(spec, args)
    try:
        common.write_json(directory / "manifest.json",
                          {"source": identity, "capture": config["capture"], "pipeline": type(pipe).__name__})
        for i, (prompt, replica) in enumerate(pending):
            seed = args.seed + replica
            sample = sample_directory(directory, prompt, seed)
            # An interrupted forced refresh must not leave an old completion marker.
            marker = sample / "complete.json"
            if marker.exists():
                marker.unlink()
            arrays = capture_one(pipe, prompt, seed, args, config, sample)
            write_npz(sample / "vectors.npz", **arrays)
            common.write_json(sample / "complete.json", {
                "schema": RAW_SCHEMA, "prompt": prompt, "seed": seed,
                "native_steps": arrays["native_steps"].tolist(),
                "scheduler_timesteps": arrays["scheduler_timesteps"].tolist(),
                "native_step_count": int(arrays["native_step_count"])})
            if i == 0:
                print(f"[grid] {type(pipe.scheduler).__name__ if hasattr(pipe, 'scheduler') else 'unknown'}: "
                      f"requested={args.num_inference_steps}, callbacks={int(arrays['native_step_count'])}, "
                      f"selected={arrays['native_steps'].tolist()}", flush=True)
            print(f"[extract] {spec['name']} sample {i + 1}/{len(pending)}", flush=True)
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return directory


def read_vectors(directory, expected_steps):
    with np.load(directory / "vectors.npz", allow_pickle=False) as data:
        names = data["names"]
        trace = np.array(data["trace"], dtype=np.float32)
        if names.ndim != 2 or names.shape[1] != 2 or trace.shape != (expected_steps, len(TRACE_NAMES)):
            raise ValueError(f"Incompatible raw vector cache: {directory}")
        vectors = {}
        for i, key in enumerate(names):
            value, valid = np.array(data[f"v{i}"]), np.array(data[f"m{i}"], dtype=bool)
            if value.ndim != 2 or value.shape[0] != expected_steps or valid.shape != (expected_steps,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid raw vectors: {directory}, {key}")
            vectors[tuple(key)] = (value, valid)
    if not np.isfinite(trace).all():
        raise ValueError("Invalid cached latent trace")
    return vectors, trace


def relation_settings(config):
    return {key: config[key] for key in ("schema", "anchors", "pools", "groups", "centering", "temporal", "min_fraction", "eps")}


def prepare_relations(raw_directory, prompts, anchors, args, config):
    destination = raw_directory / "relations" / common.digest(relation_settings(config))
    # Refresh a cached relation if any of its raw anchor/probe vectors changed.
    anchor_stamp = max((sample_directory(raw_directory, p, args.seed + r) / "vectors.npz").stat().st_mtime_ns
                       for p in anchors for r in range(args.num_images))
    def current(prompt):
        output = relation_file(destination, prompt)
        if not output.is_file():
            return False
        probe_stamp = max((sample_directory(raw_directory, prompt, args.seed + r) / "vectors.npz").stat().st_mtime_ns
                          for r in range(args.num_images))
        return output.stat().st_mtime_ns >= max(anchor_stamp, probe_stamp)
    pending = [p for p in prompts if not args.resume or not current(p)]
    if not pending:
        return destination
    steps = len(config["capture"]["steps"])
    results = {p: [] for p in pending}
    options = {"centering": config["centering"], "min_fraction": config["min_fraction"], "eps": config["eps"]}
    for r in range(args.num_images):
        anchor_vectors = [read_vectors(sample_directory(raw_directory, p, args.seed + r), steps)[0] for p in anchors]
        for prompt in pending:
            probe, trace = read_vectors(sample_directory(raw_directory, prompt, args.seed + r), steps)
            excluded = tuple(i for i, anchor in enumerate(anchors) if anchor == prompt)
            pooled, coverage = group_relations(probe, anchor_vectors, "block", steps, config["pools"],
                                               excluded=excluded, **options)
            if not np.any(coverage > 0):
                raise ValueError(f"No valid block-anchor relations for prompt {prompt!r}")
            kept = [j for j in range(len(anchors)) if j not in excluded]
            if not np.all(coverage[:, kept] == 1):
                raise ValueError("All monitored blocks must have valid anchor relations")
            lfd = lfd_relation_features(pooled, coverage, config["pools"], config["temporal"])
            means = []
            for group in config["groups"]:
                relation, fraction = group_relations(probe, anchor_vectors, group, steps, ("mean",),
                                                     excluded=excluded, **options)
                if group == "all" and not np.any(fraction > 0):
                    raise ValueError(f"No valid leaf-anchor relations for prompt {prompt!r}")
                means.append(relation["mean"])
            results[prompt].append({"lfd": lfd, "trace": trace,
                                    "module_mean": np.stack(means, axis=1)})
        del anchor_vectors
    for prompt, replicas in results.items():
        write_npz(relation_file(destination, prompt),
                  **{key: np.stack([r[key] for r in replicas]) for key in replicas[0]})
    common.write_json(destination / "manifest.json", relation_settings(config))
    return destination


def feature_names(config):
    from detection.detector import layout_names
    return layout_names(feature_layout(config))


def anchor_validity(prompt, config):
    return np.broadcast_to(np.array([a != prompt for a in config["anchors"]], bool),
                           (len(config["capture"]["steps"]), len(config["anchors"]))).copy()


def fuse_prompt(target_raw, target_relations, prompt, scorer, args, config):
    with np.load(relation_file(target_relations, prompt), allow_pickle=False) as data:
        target = {key: np.array(data[key]) for key in data.files}
    steps, count = len(config["capture"]["steps"]), len(config["anchors"])
    valid = anchor_validity(prompt, config)
    replicas = []
    for r in range(args.num_images):
        # Cached LFD is statistic-major; convert to anchor-major before fusion.
        lfd = target["lfd"][r].reshape(steps, -1, count).transpose(0, 2, 1)
        leaves = target["module_mean"][r].transpose(0, 2, 1)
        images = np.zeros((steps, count, len(config["metrics"])), np.float32)
        seed = args.seed + r
        for k, step in enumerate(config["capture"]["steps"]):
            probe_image = sample_directory(target_raw, prompt, seed) / f"step_{step:03d}.png"
            for j, anchor in enumerate(config["anchors"]):
                if valid[k, j]:
                    images[k, j] = scorer.compare(probe_image,
                        sample_directory(target_raw, anchor, seed) / f"step_{step:03d}.png")
        channels = np.concatenate([lfd, leaves, images], axis=-1)
        channels[~valid] = 0
        replicas.append(np.concatenate([channels.reshape(steps, -1), target["trace"][r]], axis=-1))
    result = np.mean(np.stack(replicas), axis=0)
    if result.shape != (steps, len(feature_names(config))) or not np.isfinite(result).all():
        raise ValueError("Invalid Stage-I feature tensor")
    return result


def extract(args, dataset):
    anchors = load_anchors(args)
    config = protocol(args, anchors)
    path = common.local_path(args.train_prompts if dataset == "train" else args.test_prompts)
    prompts = [p.strip() for p in path.read_text(encoding="utf-8-sig").splitlines() if p.strip()]
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    if not prompts or len(set(prompts)) != len(prompts):
        raise ValueError("Probe prompts must be nonempty and unique")
    specs = model_specs(args, dataset)
    identities = [source_identity(s, args) for s in specs]
    if len({common.digest(i) for i in identities}) != len(identities):
        raise ValueError("Dataset model identities must be distinct")
    if dataset == "train":
        from detection.feature_math import grouped_split
        grouped_split([{"model_id": s["name"], "label": s["label"]} for s in specs],
                      args.val_fraction, args.seed, [s.strip() for s in args.validation_models.split(",") if s.strip()])
    # Validate frozen encoders on CPU before expensive generation.
    scorer_args = argparse.Namespace(**vars(args))
    scorer_args.device = "cpu"
    scorer = common.ImageScorer(scorer_args, config["metrics"])
    target_raws = [collect_model(s, prompts, anchors, args, config) for s in specs]
    target_relations = [prepare_relations(d, prompts, anchors, args, config) for d in target_raws]
    scorer.to(args.device)
    signature_data = {"protocol": config, "encoder": scorer.identity}
    signature = common.digest(signature_data)
    output = common.local_path(args.output_dir) / f"features_{dataset}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    names = feature_names(config)
    with temp.open("w", encoding="utf-8") as stream:
        for spec, raw, relations, identity in zip(specs, target_raws, target_relations, identities):
            for i, prompt in enumerate(prompts):
                values = fuse_prompt(raw, relations, prompt, scorer, args, config)
                row = {"schema": SCHEMA, "signature": signature, "feature_names": names,
                       "model_id": spec["name"], "source_identity": common.digest(identity),
                       "architecture": spec.get("architecture", ""), "label": spec.get("label"),
                       "prompt_index": i, "original_prompt": prompt, "values": values.tolist(),
                       "layout": feature_layout(config), "anchor_valid": anchor_validity(prompt, config).tolist()}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            print(f"[fused] {spec['name']}", flush=True)
    temp.replace(output)
    common.write_json(output.with_suffix(".meta.json"), signature_data)
    print(f"Stage-I features: {output}", flush=True)


def train(args):
    from detection.detector import train as train_channels
    train_channels(args)


def predict(args):
    from detection.detector import predict as predict_channels
    predict_channels(args)
