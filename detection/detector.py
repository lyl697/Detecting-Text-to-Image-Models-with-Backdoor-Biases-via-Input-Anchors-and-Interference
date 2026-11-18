"""Stage-I classifier: anchor channels plus a separate intrinsic trajectory channel.

The JSONL stores anchor-major observations followed by trajectory observations.
Validity masks are metadata, not additional learned measurement features.
"""
import math
import numpy as np
from detection import common
from detection.common import local_path, write_json, write_csv
from detection.feature_math import grouped_split, sample_weights, TRACE_NAMES
from detection.relations import SCHEMA
from detection.protocol import feature_metadata, mismatch_report


def layout_names(layout):
    return ([f"anchor_{j:02d}_{name}" for j in range(len(layout["anchors"]))
             for name in layout["anchor_features"]]
            + [f"trajectory_{name}" for name in layout["trajectory_features"]])


def dimensions(layout):
    return len(layout["anchors"]), len(layout["anchor_features"]), len(layout["trajectory_features"])


def read_features(path, schema=SCHEMA):
    rows, observations = common.read_features(path, schema=schema)
    layout = rows[0].get("layout")
    if not isinstance(layout, dict):
        raise ValueError("Stage I requires anchor channel layout metadata")
    a, f, t = dimensions(layout)
    if a < 2 or f < 1 or layout["trajectory_features"] not in (list(TRACE_NAMES), []):
        raise ValueError("Invalid feature layout")
    if len(set(layout["anchors"])) != a or len(set(layout["anchor_features"])) != f:
        raise ValueError("Duplicate anchor or feature names")
    if rows[0]["feature_names"] != layout_names(layout):
        raise ValueError("Feature names do not match channel layout")
    masks = []
    for row in rows:
        if row.get("layout") != layout:
            raise ValueError("Mixed anchor channel layouts")
        mask = np.asarray(row.get("anchor_valid"))
        if mask.dtype != np.dtype(bool) or mask.shape != (observations.shape[1], a):
            raise ValueError("Missing/invalid anchor validity mask")
        expected = np.broadcast_to([anchor != row["original_prompt"] for anchor in layout["anchors"]], mask.shape)
        if not np.array_equal(mask, expected):
            raise ValueError("Anchor mask must exclude self comparisons")
        masks.append(mask)
    # Pack masks for batching; the network does not project them as measurements.
    return rows, np.concatenate([observations, np.asarray(masks, np.float32)], axis=-1)


def fit_normalizer(x, layout):
    a, f, t = dimensions(layout)
    n = a*f+t
    values, valid = x[..., :n].astype(np.float64), x[..., n:].astype(bool)
    weights = np.concatenate([np.repeat(valid, f, axis=-1), np.ones((*valid.shape[:-1], t), bool)], axis=-1)
    count = weights.sum(axis=(0, 1))
    mean = np.where(weights, values, 0).sum(axis=(0, 1)) / np.maximum(count, 1)
    variance = np.where(weights, (values-mean)**2, 0).sum(axis=(0, 1)) / np.maximum(count, 1)
    return mean.astype(np.float32), np.maximum(np.sqrt(variance), 1e-6).astype(np.float32)


def normalize(x, mean, std, clip, layout):
    a, f, t = dimensions(layout)
    n = a*f+t
    values = common.normalize(x[..., :n], mean, std, clip)
    valid = x[..., n:]
    channels = values[..., :a*f].reshape(*values.shape[:-1], a, f)
    channels[~valid.astype(bool)] = 0
    return np.concatenate([values, valid], axis=-1).astype(np.float32)


def build_detector(input_dim, steps, config):
    import torch
    from torch import nn
    a, f, t = dimensions(config["layout"])
    if input_dim != a*f+t+a:
        raise ValueError("Channel detector input layout mismatch")

    class AnchorChannelDetector(nn.Module):
        def __init__(self):
            super().__init__()
            width = config["hidden_dim"]
            self.anchor_project = nn.Linear(f, width)
            if t:
                self.trajectory_project = nn.Linear(t, width)
            else:
                self.summary_query = nn.Parameter(torch.zeros(1, 1, width))
            self.channel_identity = nn.Parameter(torch.empty(1, a+1, width))
            nn.init.normal_(self.channel_identity, std=.02)
            def layer():
                return nn.TransformerEncoderLayer(width, config["attention_heads"],
                    dim_feedforward=width*2, dropout=config["dropout"], activation="gelu", batch_first=True)
            self.channels = nn.TransformerEncoder(layer(), num_layers=config["channel_layers"], enable_nested_tensor=False)
            self.conv = nn.Conv1d(width, width, kernel_size=3, padding=1)
            self.position = nn.Parameter(torch.zeros(1, steps, width))
            self.temporal = nn.TransformerEncoder(layer(), num_layers=config["transformer_layers"], enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

        def forward(self, x):
            batch, time, _ = x.shape
            anchors = x[..., :a*f].reshape(batch*time, a, f)
            trace = x[..., a*f:a*f+t].reshape(batch*time, 1, t) if t else None
            valid = x[..., a*f+t:].reshape(batch*time, a) > .5
            # Invalid/self anchor observations cannot affect other tokens.
            anchors = anchors.masked_fill(~valid[..., None], 0)
            query = (torch.nn.functional.gelu(self.trajectory_project(trace)) if t else
                     self.summary_query.expand(batch*time, -1, -1))
            tokens = torch.cat([torch.nn.functional.gelu(self.anchor_project(anchors)), query], dim=1)
            padding = torch.cat([~valid, torch.zeros(batch*time, 1, dtype=torch.bool, device=x.device)], dim=1)
            encoded = self.channels(tokens + self.channel_identity, src_key_padding_mask=padding)
            # The trajectory token queries the anchors and forms each time-step summary.
            h = encoded[:, -1].reshape(batch, time, -1)
            h = h + torch.nn.functional.gelu(self.conv(h.transpose(1, 2)).transpose(1, 2))
            return self.head(self.temporal(h + self.position).mean(dim=1)).squeeze(-1)
    return AnchorChannelDetector()


def train(args, schema=SCHEMA):
    import torch
    from torch.nn import functional as F
    feature_file = local_path(args.output_dir) / "features_train.jsonl"
    rows, x = read_features(feature_file, schema=schema)
    signature_data = feature_metadata(feature_file, rows[0]["signature"])
    if signature_data is None:
        raise ValueError("Training requires the feature protocol sidecar")
    train_i, val_i = grouped_split(rows, args.val_fraction, args.seed,
                                   [s.strip() for s in args.validation_models.split(",") if s.strip()])
    split = {"strategy": "model", "seed": args.seed, "val_fraction": args.val_fraction,
             "train_models": sorted({rows[i]["model_id"] for i in train_i}),
             "validation_models": sorted({rows[i]["model_id"] for i in val_i}),
             "train_samples": len(train_i), "validation_samples": len(val_i)}
    print(f"[split] model train={split['train_models']} / {len(train_i)} samples, "
          f"validation={split['validation_models']} / {len(val_i)} samples", flush=True)
    if args.hidden_dim % args.attention_heads or args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("Invalid detector/training parameters")
    if not 0 <= args.threshold <= 1 or args.normalization_clip <= 0:
        raise ValueError("Invalid threshold or normalization clip")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    layout = rows[0]["layout"]
    mean, std = fit_normalizer(x[train_i], layout)
    x = normalize(x, mean, std, args.normalization_clip, layout)
    network = {k: getattr(args, k) for k in ("hidden_dim", "attention_heads", "transformer_layers", "dropout", "channel_layers")}
    if args.channel_layers < 1 or args.transformer_layers < 1:
        raise ValueError("Attention layer counts must be positive")
    network["layout"] = layout
    model = build_detector(x.shape[-1], x.shape[1], network).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_rows = [rows[i] for i in train_i]
    xt = torch.tensor(x[train_i])
    yt = torch.tensor([r["label"] for r in train_rows], dtype=torch.float32)
    weights = torch.tensor(sample_weights(train_rows))
    val_x = torch.tensor(x[val_i])
    val_groups = {}
    for j, i in enumerate(val_i):
        val_groups.setdefault(rows[i]["model_id"], []).append(j)
    best, stale, best_state, best_epoch = math.inf, 0, None, None
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(len(xt))
        for batch in permutation.split(args.batch_size):
            logits = model(xt[batch].to(args.device))
            losses = F.binary_cross_entropy_with_logits(logits, yt[batch].to(args.device), reduction="none")
            loss = (losses * weights[batch].to(args.device)).sum() * len(xt) / len(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            probabilities = torch.cat([model(b.to(args.device)).sigmoid().cpu() for b in val_x.split(args.batch_size)]).numpy()
        losses_by_class = {0: [], 1: []}
        for indices in val_groups.values():
            label = rows[val_i[indices[0]]]["label"]
            probability = np.clip(probabilities[indices].mean(), 1e-7, 1 - 1e-7)
            losses_by_class[label].append(-label * np.log(probability) - (1-label) * np.log(1-probability))
        validation_loss = float(np.mean([np.mean(v) for v in losses_by_class.values()]))
        if not math.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss; no checkpoint will be saved")
        print(f"[train] epoch={epoch+1} model_validation_loss={validation_loss:.6f}", flush=True)
        if validation_loss < best:
            best, stale, best_epoch = validation_loss, 0, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = {"schema": schema, "signature": rows[0]["signature"], "feature_names": rows[0]["feature_names"],
                  "network": network, "steps": x.shape[1], "state_dict": best_state, "split": split,
                  "mean": mean.tolist(), "std": std.tolist(), "normalization_clip": args.normalization_clip,
                  "threshold": args.threshold, "best_epoch": best_epoch, "validation_loss": best,
                  "train_models": sorted({rows[i]["model_id"] for i in train_i}),
                  "validation_models": sorted({rows[i]["model_id"] for i in val_i}),
                  "training_sources": sorted({r["source_identity"] for r in rows}),
                  "training_args": vars(args)}
    if signature_data is not None:
        checkpoint["signature_data"] = signature_data
    output = local_path(args.checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(output)
    write_json(output.with_suffix(".json"), {k: v for k, v in checkpoint.items() if k != "state_dict"})
    print(f"Detector: {output}", flush=True)


def predict(args, schema=SCHEMA, *, allow_training_overlap=True):
    import torch
    from detection.evaluation import aggregate_rows
    checkpoint = torch.load(local_path(args.checkpoint), map_location="cpu", weights_only=True)
    feature_file = local_path(args.output_dir) / "features_test.jsonl"
    rows, x = read_features(feature_file, schema=schema)
    if checkpoint["schema"] != schema or checkpoint["signature"] != rows[0]["signature"]:
        raise ValueError(mismatch_report(checkpoint, local_path(args.checkpoint), feature_file, rows[0]))
    if checkpoint["feature_names"] != rows[0]["feature_names"] or checkpoint["steps"] != x.shape[1]:
        raise ValueError("Checkpoint feature layout mismatch")
    if not allow_training_overlap and any(r["source_identity"] in checkpoint["training_sources"] for r in rows):
        raise ValueError("Test checkpoint overlaps detector training/validation models")
    layout = checkpoint["network"]["layout"]
    if rows[0]["layout"] != layout:
        raise ValueError("Checkpoint channel layout mismatch")
    x = normalize(x, np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"]), checkpoint["normalization_clip"], layout)
    model = build_detector(x.shape[-1], x.shape[1], checkpoint["network"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        probabilities = torch.cat([model(b.to(args.device)).sigmoid().cpu()
                                   for b in torch.tensor(x).split(args.batch_size)]).numpy()
    output = []
    for row, probability in zip(rows, probabilities):
        output.append({"model_id": row["model_id"], "architecture": row["architecture"],
                       "prompt_index": row["prompt_index"], "original_prompt": row["original_prompt"],
                       "true_label": "" if row["label"] is None else row["label"],
                       "model_tag": row["model_id"], "predicted_label": int(probability >= .5),
                       "backdoor_probability": float(probability)})
    directory = local_path(args.output_dir)
    write_csv(directory / "prompt_predictions.csv", output)
    write_csv(directory / "stage1.csv", aggregate_rows(output, checkpoint["threshold"]))
    print(f"Prompt predictions: {directory / 'prompt_predictions.csv'}")
    print(f"Stage-I decisions: {directory / 'stage1.csv'} (threshold from checkpoint)")
