"""Latent dynamics, model splits, and training weights."""
import numpy as np

TRACE_NAMES = ("cosine", "relative_rms", "log_rms", "relative_update", "transition_valid")

def latent_trajectory(states, eps=1e-12):
    """Adjacent selected POST-step latents; no cross-model coordinates."""
    z = [np.asarray(x, dtype=np.float64).reshape(-1) for x in states]
    if not z or any(not x.size or not np.isfinite(x).all() for x in z):
        raise ValueError("Latents must be nonempty and finite")
    if len({x.size for x in z}) != 1:
        raise ValueError("Latent shape changed within one model trajectory")
    norms = [np.linalg.norm(x) for x in z]
    rms = np.array(norms) / np.sqrt(z[0].size)
    result = []
    for i, x in enumerate(z):
        valid = i > 0 and norms[i - 1] > eps and norms[i] > eps
        cosine = np.clip(np.dot(z[i-1], x) / (norms[i-1] * norms[i]), -1, 1) if valid else 0.
        update = np.linalg.norm(x - z[i-1]) / max(norms[i-1], eps) if valid else 0.
        result.append([cosine, rms[i] / max(rms[0], eps), np.log1p(rms[i]), update, float(valid)])
    return np.asarray(result, dtype=np.float32)


def grouped_split(rows, val_fraction=.25, seed=42, validation_models=()):
    """Model-disjoint split stratified by label, never by prompt."""
    labels = {}
    for row in rows:
        name, label = row["model_id"], row["label"]
        if label not in (0, 1):
            raise ValueError("Training labels must be 0 or 1")
        if name in labels and labels[name] != label:
            raise ValueError("Conflicting labels for one model")
        labels[name] = label
    if set(labels.values()) != {0, 1}:
        raise ValueError("Both clean and backdoor models are required")
    if validation_models:
        val = set(validation_models)
        if not val <= labels.keys():
            raise ValueError("Unknown validation model name")
    else:
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")
        rng = np.random.default_rng(seed)
        val = set()
        for label in (0, 1):
            names = sorted(k for k, v in labels.items() if v == label)
            if len(names) < 2:
                raise ValueError("Need >=2 distinct models per class for model-disjoint validation")
            rng.shuffle(names)
            count = min(len(names) - 1, max(1, round(len(names) * val_fraction)))
            val.update(names[:count])
    train = set(labels) - val
    if {labels[n] for n in train} != {0, 1} or {labels[n] for n in val} != {0, 1}:
        raise ValueError("Both train and validation must contain both classes")
    return ([i for i, r in enumerate(rows) if r["model_id"] in train],
            [i for i, r in enumerate(rows) if r["model_id"] in val])


def sample_weights(rows):
    """Equal mass per class, then model, then prompt."""
    counts, labels = {}, {}
    for r in rows:
        k = r["model_id"]
        counts[k] = counts.get(k, 0) + 1
        labels[k] = r["label"]
    class_counts = {c: sum(v == c for v in labels.values()) for c in (0, 1)}
    if not all(class_counts.values()):
        raise ValueError("Both classes required")
    return np.array([1. / (2 * class_counts[r["label"]] * counts[r["model_id"]])
                     for r in rows], dtype=np.float32)


def normalize(x, mean, std, clip=10.):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Features contain nonfinite values")
    return np.clip((x - mean) / std, -clip, clip).astype(np.float32)
