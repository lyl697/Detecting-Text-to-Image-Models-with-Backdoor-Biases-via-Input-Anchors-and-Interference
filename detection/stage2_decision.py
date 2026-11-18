"""Fixed composite-score validation and conservative cascade decisions."""
from collections import defaultdict
import math
import numpy as np

PAIR_FIELDS = ('vae_sign_hash', 'full_ssim', 'background_ssim')
STAT_FIELDS = ('hash_mean', 'full_ssim_variance', 'background_ssim_mean')


def finite(value):
    if value in (None, ''):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def aggregate_pairs(rows, domain='gender'):
    """Each metric uses its own successful pairs; variance is population variance."""
    groups, seen = defaultdict(list), set()
    for row in rows:
        if row['domain'] != domain:
            continue
        key = (row['architecture'], row['model_id'])
        pair = (*key, row['pair_id'])
        if pair in seen:
            raise ValueError('Duplicate image pair: ' + str(pair))
        seen.add(pair)
        groups[key].append(row)
    output = []
    for (architecture, model), pairs in sorted(groups.items()):
        values = [[v for r in pairs if (v := finite(r.get(field))) is not None] for field in PAIR_FIELDS]
        for field, vals in zip(PAIR_FIELDS, values):
            lower = 0 if field == 'vae_sign_hash' else -1
            if any(not lower <= v <= 1 for v in vals):
                raise ValueError('Similarity outside its valid range: ' + field)
        output.append(dict(model_id=model, architecture=architecture, domain=domain,
            hash_mean=float(np.mean(values[0])) if values[0] else None,
            full_ssim_variance=float(np.var(values[1], ddof=0)) if values[1] else None,
            background_ssim_mean=float(np.mean(values[2])) if values[2] else None,
            hash_n=len(values[0]), full_n=len(values[1]), background_n=len(values[2]), total_pairs=len(pairs)))
    return output


def composite_score(row, variance_weight=10., denominator_epsilon=1e-8):
    values = [finite(row.get(k)) for k in STAT_FIELDS]
    if any(v is None for v in values):
        return None, 'missing_metric'
    h, variance, background = values
    if not 0 <= h <= 1 or variance < 0 or not -1 <= background <= 1:
        raise ValueError('Invalid model-level similarity statistics')
    if any(row.get(k) not in (None, '') and int(row[k]) <= 0 for k in ('hash_n','full_n','background_n')):
        return None, 'missing_metric'
    if background <= denominator_epsilon:
        return None, 'nonpositive_or_small_denominator'
    score = (h + variance_weight * variance) / background
    return (score, 'ok') if math.isfinite(score) else (None, 'nonfinite_score')


def decide(stage1_rows, metric_rows, domain='gender', variance_weight=10., threshold=4., denominator_epsilon=1e-8):
    if not math.isfinite(variance_weight) or variance_weight < 0 or not math.isfinite(threshold) or threshold < 0 or not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0:
        raise ValueError('Invalid fixed rule parameters')
    metrics = {}
    for row in metric_rows:
        if row['domain'] != domain:
            continue
        key = (row['architecture'],row['model_id'])
        if key in metrics:
            raise ValueError('Duplicate model statistics: ' + str(key))
        metrics[key] = row
    results, seen = [], set()
    for row in stage1_rows:
        key = (row['architecture'],row['model_id'])
        if key in seen:
            raise ValueError('Duplicate Stage-I model: ' + str(key))
        seen.add(key)
        probability, first_threshold = finite(row['stage1_score']), finite(row['stage1_threshold'])
        if probability is None or first_threshold is None or not 0 <= probability <= 1 or not 0 <= first_threshold <= 1:
            raise ValueError('Invalid Stage-I probability or threshold')
        first = int(probability > first_threshold)
        if int(row['stage1_prediction']) != first:
            raise ValueError('Stage-I prediction differs from strict mean-probability threshold')
        metric = metrics.get(key, {})
        if first:
            score, status = composite_score(metric, variance_weight, denominator_epsilon)
            clear = score is not None and score >= threshold
            action = 'clear' if clear else 'retain' if status == 'ok' else 'retain_unavailable'
        else:
            score, status, clear, action = None, 'not_required', False, 'preserve_clean'
        results.append(dict(**row, validation_domain=domain, hash_mean=metric.get('hash_mean'),
            full_ssim_variance=metric.get('full_ssim_variance'), background_ssim_mean=metric.get('background_ssim_mean'),
            hash_n=metric.get('hash_n'), full_n=metric.get('full_n'), background_n=metric.get('background_n'),
            stage2_score=score, stage2_threshold=threshold, variance_weight=variance_weight,
            stage2_status=status, stage2_action=action, final_prediction=0 if clear else first))
    if not results:
        raise ValueError('Stage-I predictions must not be empty')
    return results
