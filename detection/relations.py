"""Within-model activation relations; no cross-model layer alignment."""
import numpy as np

SCHEMA = "paper_stage1_v1"
RAW_SCHEMA = "paper_raw_v1"

def cosine_to_anchors(probe, anchors, probe_valid=None, anchor_valid=None,
                      centering="none", excluded=(), eps=1e-12):
    """probe [K,D], anchors [A,K,D]. Return similarities/mask [K,A]."""
    p = np.asarray(probe, dtype=np.float64)
    a = np.asarray(anchors, dtype=np.float64)
    if p.ndim != 2 or a.ndim != 3 or a.shape[1:] != p.shape or not a.shape[0]:
        raise ValueError("Probe and anchors must share a module's shape within one model")
    if not np.isfinite(p).all() or not np.isfinite(a).all() or eps <= 0:
        raise ValueError("Nonfinite activation or invalid relation epsilon")
    pv = np.ones(p.shape[0], bool) if probe_valid is None else np.asarray(probe_valid, dtype=bool)
    av = np.ones(a.shape[:2], bool) if anchor_valid is None else np.asarray(anchor_valid, dtype=bool).copy()
    if pv.shape != p.shape[:1] or av.shape != a.shape[:2]:
        raise ValueError("Invalid per-step validity masks")
    if excluded:
        av[list(excluded)] = False
    if centering != "none":
        raise ValueError("Activation comparisons require uncentered cosine relations")
    pn = np.linalg.norm(p, axis=-1)
    an = np.linalg.norm(a, axis=-1)
    valid = av & pv[None] & (pn[None] > eps) & (an > eps)
    dots = np.einsum("kd,akd->ak", p, a)
    values = np.zeros_like(dots)
    np.divide(dots, an * pn[None], out=values, where=valid)
    return np.clip(values.T, -1., 1.).astype(np.float32), valid.T


def pool_relations(values, masks, pools=("mean", "std"), min_fraction=.5):
    """Pool [L,K,A] without layer alignment or padding-as-data."""
    x, mask = np.asarray(values, dtype=np.float64), np.asarray(masks, dtype=bool)
    if x.ndim != 3 or x.shape != mask.shape or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite layer/step/anchor relations and matching masks")
    if not 0 <= min_fraction <= 1 or not pools or not set(pools) <= {"mean", "std"}:
        raise ValueError("Invalid relation pooling configuration")
    counts = mask.sum(axis=0)
    fraction = counts / len(x)
    valid = (counts > 0) & (fraction >= min_fraction)
    mean = np.where(mask, x, 0).sum(axis=0) / np.maximum(counts, 1)
    output = {}
    for name in pools:
        if name == "mean":
            item = mean.copy()
        elif name == "std":
            item = np.sqrt(np.where(mask, (x-mean)**2, 0).sum(axis=0) / np.maximum(counts, 1))
        output[name] = np.where(valid, item, 0).astype(np.float32)
    return output, np.where(valid, fraction, 0).astype(np.float32)


def group_relations(probe, anchors, group, steps, pools=("mean", "std"),
                    centering="none", min_fraction=.5, excluded=(), eps=1e-12):
    """Dictionaries map (kind, module_name) to (values[K,D], valid[K])."""
    keys = [key for key in probe if (key[0] != "block" if group == "all" else key[0] == group)]
    if not keys:
        return ({name: np.zeros((steps, len(anchors)), np.float32) for name in pools},
                np.zeros((steps, len(anchors)), np.float32))
    values, masks = [], []
    for key in keys:
        vector, valid = probe[key]
        matrix, available = [], []
        for anchor in anchors:
            a, v = anchor.get(key, (np.zeros_like(vector), np.zeros(steps, bool)))
            if a.shape != vector.shape:
                raise ValueError(f"Channel width changed within the same model for {key}")
            matrix.append(a)
            available.append(v)
        cos, mask = cosine_to_anchors(vector, np.stack(matrix), valid, np.stack(available),
                                     centering, excluded, eps)
        values.append(cos)
        masks.append(mask)
    return pool_relations(values, masks, pools, min_fraction)


def lfd_relation_features(pooled, coverage, pool_names=("mean", "std"), temporal=True):
    """Only mean, population standard deviation, and signed temporal difference."""
    if tuple(pool_names) != ("mean", "std") or not temporal:
        raise ValueError("Block features require mean, standard deviation, and temporal difference")
    mean = pooled["mean"]
    delta = np.zeros_like(mean)
    # Extraction rejects invalid monitored blocks before reaching this function.
    delta[1:] = mean[1:] - mean[:-1]
    return np.concatenate([mean, pooled["std"], delta], axis=-1).astype(np.float32)
