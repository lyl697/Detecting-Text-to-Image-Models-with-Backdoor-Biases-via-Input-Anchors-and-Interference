# Detecting Backdoor Biases in Text-to-Image Diffusion Models

This repository implements a two-stage framework for detecting backdoor biases in text-to-image diffusion models, supporting Stable Diffusion 1.4, Stable Diffusion 2, Stable Diffusion 3.5, and FLUX.1.

Stage I compares hidden activations and intermediate images against semantic anchors, combines these relationships with latent-trajectory features, and learns a model-level detector. Stage II validates suspicious models through attribute perturbations and a composite image-similarity score. The repository also provides experiments for feature-group contributions, anchor counts, capture schedules, and prompt budgets.

## Installation and assets

Use Python 3.10 or newer. Install a CUDA-compatible PyTorch build for your machine, followed by:

```bash
python -m pip install -r requirements.txt
```

The pinned versions are from the experiment environment. Prepare the pretrained model weights and dependencies before running on an offline server. LPIPS also requires the pretrained AlexNet weights in the Torch cache.

Set the model paths in `configs/sd14.json`, `configs/sd2.json`, `configs/sd35.json`, or `configs/flux.json`. Each file contains six source models and forty evaluation models, with identifiers used in output files and directories. Paths are relative to this repository. SD1.4/SD2 entries use complete Diffusers checkpoints; SD3.5/FLUX entries use a shared base checkpoint and an optional LoRA directory. An empty `lora_id` selects the base checkpoint. Keep the same checkpoint path wherever a model appears in both lists.

Required shared assets:

| Asset | Default location |
| --- | --- |
| CLIP ViT-L/14 image encoder, including projection and image processor | `models/clip-vit-large-patch14/` |
| Common SD2 VAE for **all** Stage-II architectures | `models/sd2/base/vae/` |
| YOLO segmentation checkpoint | `models/yolo26x-seg.pt` |

Train/test labels are model labels. They are used for supervised Stage-I training and evaluation, never to choose Stage-II perturbations or to change the fixed validation rule. Training and validation are split by model identity: `backdoor_1,clean_3` are validation models, and the other four source models are training models. Evaluation inventories preserve the original train/validation overlaps. Summaries report both the complete inventory and the subset excluding source-model overlap; the former is not an entirely unseen-model test set.

## Full pipeline

All commands below run from this directory. GPU selection uses the physical CUDA index; the process sees the selected device as `cuda:0`.

```bash
python run_experiment.py --architecture sd2 --check
CUDA_VISIBLE_DEVICES=1 python run_experiment.py --architecture sd2
```

Replace `sd2` with `sd14`, `sd35`, or `flux`. The runner extracts source features, trains the classifier, evaluates all configured models, and validates Stage-I positives with Stage II. Each evaluation model runs in a separate subprocess. The runner reuses completed compatible features, training, predictions, and Stage-II images. Changed protocols require a new `--output` directory.

Run either stage separately:

```bash
CUDA_VISIBLE_DEVICES=1 python run_experiment.py --architecture sd35 --stages stage1
CUDA_VISIBLE_DEVICES=1 python run_experiment.py --architecture sd35 --stages stage2
```

To test a subset, add `--only sd2_model_01 sd2_model_02`. Previously completed models remain in the aggregate summary. A failed extraction is reported; it is not silently counted as a successful model. Stage-II measurement failures retain the corresponding Stage-I decision and are reported as unavailable.

## Stage I

Feature files and classifier checkpoints record their extraction protocol. Training and testing must use matching anchors, capture schedules, feature layouts, and image-encoder weights. Extract the features before training a corresponding classifier.

The editable defaults are in `detection/settings.py`. Main settings:

| Setting | Value |
| --- | --- |
| Generation inference steps | 50 |
| Captured zero-based callback indices | `0,2,5,7,10,12,15,17,20,22,25,27,30,32,35,37,40,42,45,47,49` |
| Anchor prompts | First 7 entries of `data/anchors.txt` |
| Probe prompts | 100 from `data/train.txt` or `data/test.txt` |
| Generation seed / replicas | 42 / 1; same seed for anchors and probes |
| Resolution | 768 × 768 |
| Guidance | 7.5 for SD1.4/SD2/SD3.5; 4.5 for FLUX |
| Generation precision | FP16; FLUX BF16 |
| Model-level threshold | Strict mean probability **> 0.5** |

The sampler still performs 50 requested inference steps. Callback indices are observation positions, not scheduler noise values. Schedulers with startup substeps can expose more callbacks; the fixed index grid is preserved and the actual scheduler timesteps are saved.

Per anchor and captured time, the ten numerical features are:

1. Block cosine-relation mean, population standard deviation, and signed difference between adjacent **selected** times.
2. Mean cosine relation over supported leaf modules (all modules receive equal weight).
3. CLIP similarity, LPIPS distance, RGB MSE, SSIM, HOG-descriptor-histogram Pearson correlation, and DTW distance.

Five intrinsic latent-trajectory measurements are added once per time: cosine to the previous selected latent, RMS relative to the first selected latent, `log1p(RMS)`, relative update magnitude, and transition validity. Thus the default numerical tensor is **21 × 75 = 21 × (7 × 10 + 5)**. All monitored blocks must have valid relations; unexpected missing/zero-norm block observations raise an error. The first block temporal difference is explicitly zero. Later differences are computed normally and may naturally also equal zero. Self-anchor comparisons are excluded with an attention mask; these masks are not learned numerical features.

The shared anchor projection and separate trajectory projection produce 64-dimensional tokens. A one-layer, four-head channel Transformer summarizes each time using the trajectory token. A residual temporal Conv1d (kernel 3) precedes a one-layer temporal Transformer. Time-mean pooling, LayerNorm, and a linear head produce the prompt probability. Dropout is 0.1. Without the trajectory group, the ablation uses a learned sample-independent summary token.

Normalization uses only training-model records, pooling across records and captured times per coordinate, with a standard-deviation floor of `1e-6` and clipping at ±10. Training uses class/model-balanced BCE, AdamW (`lr=0.001`, `weight_decay=0.0001`), batch size 32, at most 100 epochs, and early stopping patience 15. The best checkpoint minimizes class-balanced model-level validation loss. A prompt's label denotes its generating model, not whether that particular prompt triggered an attack.

Direct entry points:

```bash
python stage1.py --architecture sd2 --mode check
CUDA_VISIBLE_DEVICES=1 python stage1.py --architecture sd2 --mode extract --dataset train
CUDA_VISIBLE_DEVICES=1 python stage1.py --architecture sd2 --mode train
CUDA_VISIBLE_DEVICES=1 python stage1.py --architecture sd2 --mode extract --dataset test \
  --target_model models/sd2/model_01 --model_name sd2_model_01 --output_dir outputs/sd2/single
CUDA_VISIBLE_DEVICES=1 python stage1.py --architecture sd2 --mode predict \
  --output_dir outputs/sd2/single --checkpoint outputs/sd2/stage1/detector.pt
```

## Stage II

The default evaluation domain is **gender**, fixed for every model. Doctor and teacher each have five original prompts. Append `The person is male.` or `The person is female.`. One reference image and four independently seeded images per target give 90 images and 80 comparisons per model. Seeds of the reference and both target groups are disjoint. Other predefined domains are available through `--domain`, but are separate evaluations; no domain is chosen according to model labels or measured scores, and there is no implicit six-domain aggregation.

Stage-II generation uses the following settings:

| Architecture | Inference steps | Guidance | Precision | Resolution |
| --- | ---: | ---: | --- | --- |
| SD1.4 / SD2 | 30 | 7.5 | FP16 | 768 × 768 |
| SD3.5 | 20 | 4.5 | BF16 | 768 × 768 |
| FLUX | 20 | 3.5 | BF16 | 768 × 768 |

All models use the same FP32 VAE encoder for measurement. Images are resized to 256 × 256 with bicubic interpolation, encoded to the posterior mode **before diffusion scaling**, and compared by the fraction of equal coordinate signs at zero. Full-image SSIM uses a 256 × 256 Lanczos resize and normalized RGB luminance. The segmenter identifies persons at size 640; instance masks are unioned and resized to a 448 × 448 grid. Background excludes a foreground dilation of 8 pixels. Zero detections uses the full image as background, matching the measurement protocol, and this fallback is recorded. Background comparison uses mask-aware bilinear resizing, at least 10% common-background coverage, and complete 11 × 11 SSIM windows. Gaussian sigma is 1.5, range is 1, and covariance uses population normalization.

For the selected domain, pool both categories and both target attributes. Each successful finite image pair has equal weight; each background pair contributes one window-averaged score. Each metric uses its own available pairs. Define:

```text
S = (mean(VAE spatial sign-hash similarity)
     + 10 * population_variance(full-image SSIM))
    / mean(common-background SSIM)
```

Only Stage-I positives are considered for revision. If `S >= 4`, change the prediction to clean; otherwise retain the positive prediction. Stage-I negatives remain clean. Missing metrics, nonfinite values, or a denominator ≤ `1e-8` retain the Stage-I prediction. Counts and status are exported. Labels never enter the score or cascade function. The script does not fit a threshold.

The variance coefficient 10 and decision threshold 4 were selected using development results and remain fixed during evaluation. Independent validation requires a separate test cohort evaluated with these same constants.

```bash
python stage2.py --architecture sd2 --mode check
CUDA_VISIBLE_DEVICES=1 python stage2.py --architecture sd2 --mode all \
  --stage1_csv outputs/sd2/stage1/stage1.csv
```

Apply the rule to previously measured **wide** statistics without loading GPU models:

```bash
python stage2.py --mode decide --architecture sd2 \
  --stage1_csv outputs/sd2/stage1/stage1.csv \
  --metrics model_metrics.csv --output_dir outputs/sd2/validation
```

`model_metrics.csv` columns: `model_id,architecture,domain,hash_mean,full_ssim_variance,background_ssim_mean,hash_n,full_n,background_n,total_pairs`. Model identifiers must match those in the Stage-I inference output `stage1.csv`.

## Ablation studies

All ablations use the block-relation mean, standard deviation, and temporal mean difference as the hidden-block feature group.

```bash
# Nested 7/14/21/28/35 anchors, fixed six capture times; refit every detector.
CUDA_VISIBLE_DEVICES=1 python experiments.py --mode anchors --architecture sd2

# 6/11/21/50 capture times, fixed seven anchors and 50 sampling steps.
CUDA_VISIBLE_DEVICES=1 python experiments.py --mode steps --architecture flux

# Generate the six-capture baseline used by feature/prompt ablations.
CUDA_VISIBLE_DEVICES=1 python run_experiment.py --architecture sd2 --stages stage1 \
  --step_count 6 --output outputs/sd2_steps6

# Full configuration plus removal of each of the four groups; retrain each.
CUDA_VISIBLE_DEVICES=1 python experiments.py --mode features --architecture sd2 \
  --source outputs/sd2_steps6/stage1

# Fixed classifier; nested global random subsets from all 100 prompts, 100 repeats.
python experiments.py --mode prompts --architecture sd2 \
  --source outputs/sd2_steps6/stage1 --counts 10,20,50,100 --repeats 100
```

Anchor/step sweeps support `--counts` and `--check`. Prompt subsets use seed 42; all models share the selected subset within a repeat. Error bars describe subset variability, not training or generation randomness. `N=100` is evaluated once. Feature ablations preserve the original source split and training settings. Missing test feature files are reported and not silently included in denominators.

## Outputs and validation

The main outputs are `stage1/features_train.jsonl`, `stage1/detector.pt`, `stage1/batch_test/<id>/features_test.jsonl`, per-prompt CSVs, aggregate `stage1/stage1.csv`, and `stage2/gender/final_predictions.csv`. JSON summaries include confusion counts and source-overlap exclusions. Raw captures and measurement caches are local runtime artifacts, excluded from version control.

```bash
python -B -m unittest discover -s tests -v
```

The included tests cover feature contracts, model-level decisions, perturbation seeds, measurements, and conservative validation. Tests requiring PyTorch are skipped if it is unavailable. Reproducing detection accuracy requires running the full pipeline with the model weights and evaluation data. Comparison baselines use their respective external implementations.
