"""Strict feature/checkpoint protocol comparisons."""
import json
from pathlib import Path
from detection import common

def differences(expected, actual, path=""):
    """Exact differences; anchor positions are one-based for readable logs."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(set(expected) | set(actual)):
            name = f"{path}.{key}" if path else key
            if key not in expected:
                result.append(f"{name}: expected=<absent>; actual={actual[key]!r}")
            elif key not in actual:
                result.append(f"{name}: expected={expected[key]!r}; actual=<absent>")
            else:
                result.extend(differences(expected[key], actual[key], name))
        return result
    if type(expected) is type(actual) and expected == actual:
        return []
    if path.endswith("anchors") and isinstance(expected, list) and isinstance(actual, list):
        result = []
        for i in range(max(len(expected), len(actual))):
            left = expected[i] if i < len(expected) else "<absent>"
            right = actual[i] if i < len(actual) else "<absent>"
            if left != right:
                result.append(f"{path}[{i + 1}]: expected={left!r}; actual={right!r}")
        return result
    return [f"{path}: expected={expected!r}; actual={actual!r}"]


def feature_metadata(path, signature):
    """Never trust a stale sidecar whose digest does not match the JSONL."""
    meta_path = Path(path).with_suffix(".meta.json")
    if not meta_path.is_file():
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or "protocol" not in data or "encoder" not in data:
        raise ValueError(f"Invalid extraction metadata: {meta_path}")
    if common.digest(data) != signature:
        raise ValueError(f"Metadata digest does not match feature signature: {meta_path}")
    return data


def mismatch_report(checkpoint, checkpoint_path, feature_file, row):
    lines = [
        "Checkpoint and test extraction protocol/encoder mismatch.",
        f"  checkpoint: {checkpoint_path}",
        f"  features: {feature_file}",
        f"  checkpoint signature: {checkpoint.get('signature')}",
        f"  feature signature:    {row.get('signature')}",
        "  expected = checkpoint training protocol; actual = saved test features.",
    ]
    expected, notes = checkpoint.get("signature_data"), []
    if expected is None or common.digest(expected) != checkpoint.get("signature"):
        expected = None
        notes.append("Checkpoint extraction metadata is missing or invalid.")
    try:
        actual = feature_metadata(feature_file, row["signature"])
    except (OSError, ValueError) as exc:
        actual = None
        notes.append(str(exc))
    diff = differences(checkpoint.get("schema"), row.get("schema"), "schema")
    if expected is not None and actual is not None:
        diff.extend(differences(expected, actual))
    else:
        diff.extend(differences(checkpoint.get("network", {}).get("layout", {}).get("anchors"),
                                row.get("layout", {}).get("anchors"), "layout.anchors"))
        if actual is None:
            notes.append("Test extraction metadata missing or invalid; exact step/encoder differences unavailable.")
    lines.extend("  " + item for item in diff + notes)
    lines.append("Use one extraction protocol for training and testing. "
                 "Re-extract incompatible features in a new output directory; "
                 "retrain if training features change.")
    return "\n".join(lines)
