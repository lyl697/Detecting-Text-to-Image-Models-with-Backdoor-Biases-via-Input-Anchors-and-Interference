"""Model-mean decisions and explicitly labeled evaluation."""
from collections import defaultdict
import csv
from pathlib import Path
MODEL_KEYS = ("model_id", "model_tag", "name")


def metrics(rows, prediction='stage1_prediction'):
    labeled = [r for r in rows if str(r.get('true_label', '')) in ('0', '1')]
    counts = {key:0 for key in ('tp','tn','fp','fn')}
    for row in labeled:
        truth, pred = int(row['true_label']), int(row[prediction])
        counts['tp' if truth and pred else 'fn' if truth else 'fp' if pred else 'tn'] += 1
    tp, tn, fp, fn = (counts[k] for k in ('tp','tn','fp','fn'))
    divide = lambda x,y: x/y if y else None
    return dict(labeled_models=len(labeled), **counts, accuracy=divide(tp+tn,len(labeled)),
                tpr=divide(tp,tp+fn), fpr=divide(fp,fp+tn), f1=divide(2*tp,2*tp+fp+fn))


def summaries(rows, prediction='stage1_prediction'):
    independent = [r for r in rows if str(r.get('train_validation_overlap', 'unknown')) == 'none']
    return dict(all_labeled=metrics(rows,prediction), excluding_train_validation_overlap=metrics(independent,prediction))

def _model_id(row: dict) -> str:
    for key in MODEL_KEYS:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise ValueError("Each prediction row needs model_id or model_tag")


def aggregate_rows(
    rows: list[dict],
    threshold: float,
) -> list[dict]:
    """Return one Stage-I row per model using a strict ``score > threshold``."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("architecture", "")).strip(), _model_id(row))].append(row)

    output = []
    for architecture_key, model_id in sorted(grouped):
        model_rows = grouped[architecture_key, model_id]
        probabilities = []
        prompt_indices = set()
        for row in model_rows:
            try:
                probability = float(row["backdoor_probability"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{model_id}: invalid backdoor_probability in row {row!r}"
                ) from exc
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{model_id}: probability outside [0, 1]: {probability}")
            probabilities.append(probability)
            prompt_index = str(row.get("prompt_index", "")).strip()
            if not prompt_index:
                raise ValueError(f"{model_id}: missing prompt_index")
            if prompt_index in prompt_indices:
                raise ValueError(f"{model_id}: duplicate prompt_index {prompt_index}")
            prompt_indices.add(prompt_index)

        score = sum(probabilities) / len(probabilities)
        architectures = {str(r.get("architecture", "")).strip() for r in model_rows} - {""}
        labels = {str(r.get("true_label", "")).strip() for r in model_rows} - {""}
        if len(architectures) > 1 or len(labels) > 1:
            raise ValueError(f"{model_id}: inconsistent architecture or true_label")
        output.append({
            "model_id": model_id,
            "architecture": next(iter(architectures), ""),
            "true_label": next(iter(labels), ""),
            "num_prompts": len(prompt_indices),
            "stage1_score": score,
            "stage1_threshold": float(threshold),
            "stage1_prediction": int(score > threshold),
        })
    return output


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))
