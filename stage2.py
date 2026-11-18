"""Generate interventions, measure three similarities, and apply the fixed Stage-II rule."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from detection import common
from detection.settings import load_config
from detection.stage2_design import SPECS, plan
from detection.stage2_features import ImageFeatures, full_ssim, background_ssim, sign_hash
from detection.stage2_decision import aggregate_pairs, decide
from detection.evaluation import read_csv, summaries


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode', choices=['check','generate','measure','decide','all'], default='check')
    p.add_argument('--architecture', choices=['sd14','sd2','sd35','flux'], default='sd2')
    p.add_argument('--models_config', default=None)
    p.add_argument('--only', nargs='+', help='Model identifiers from the model configuration')
    p.add_argument('--stage1_csv', default=None, help='Actual Stage-I output; mandatory for all/decide')
    p.add_argument('--metrics', default=None, help='Optional precomputed wide model_metrics.csv')
    p.add_argument('--output_dir', default=None)
    p.add_argument('--domain', choices=list(SPECS), default='gender')
    p.add_argument('--prompt_file', default='data/stage2_prompts.json')
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=None, help='SD1.4/SD2: 30; SD3.5/FLUX: 20')
    p.add_argument('--guidance_scale', type=float, default=None, help='SD1.4/SD2: 7.5; SD3.5: 4.5; FLUX: 3.5')
    p.add_argument('--height', type=int, default=768)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--vae_model', default='models/sd2/base/vae', help='Same pretrained VAE for every architecture')
    p.add_argument('--segmenter_weights', default='models/yolo26x-seg.pt')
    p.add_argument('--vae_size', type=int, default=256)
    p.add_argument('--pixel_size', type=int, default=256)
    p.add_argument('--segment_size', type=int, default=640)
    p.add_argument('--mask_size', type=int, default=448)
    p.add_argument('--mask_margin', type=int, default=8)
    p.add_argument('--min_common_background', type=float, default=.1)
    p.add_argument('--variance_weight', type=float, default=10.)
    p.add_argument('--threshold', type=float, default=4.)
    p.add_argument('--denominator_epsilon', type=float, default=1e-8)
    return p


def read_manifest(folder):
    return json.loads((folder/'generation.json').read_text(encoding='utf-8'))


def write_protocol(path, value):
    value = json.loads(json.dumps(value))
    if path.is_file() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise ValueError(f'Protocol changed: {path}. Choose a new output directory.')
    common.write_json(path, value)


def image_valid(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (OSError, ValueError):
        return False


def generate(args, model, folder):
    common.write_json(folder/'incomplete.json', dict(stage='generation'))
    import torch
    manifest = plan(args, model)
    options = SimpleNamespace(device=args.device, dtype='float32' if args.device=='cpu' else
        'bfloat16' if args.architecture in ('sd35','flux') else 'float16', local_files_only=True, revision='main')
    manifest.update(source=common.source_identity(model, options), generation=dict(
        steps=args.steps, guidance_scale=args.guidance_scale, height=args.height, width=args.width, dtype=options.dtype))
    write_protocol(folder/'generation.json', manifest)
    pending = [r for r in manifest['images'] if not image_valid(folder/r['path'])]
    if not pending:
        print('[skip] complete Stage-II images:',model['name'])
        return
    pipe = common.load_pipeline(model, options)
    if hasattr(pipe, 'safety_checker'):
        pipe.safety_checker = None
    try:
        for i,row in enumerate(pending):
            with torch.inference_mode():
                image = pipe(prompt=row['prompt'], generator=torch.Generator(device=args.device).manual_seed(row['seed']),
                    num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                    height=args.height, width=args.width).images[0]
            path = folder/row['path']; path.parent.mkdir(parents=True,exist_ok=True)
            temp = path.with_suffix('.tmp.png'); image.save(temp); temp.replace(path)
            print(f"[Stage II] {model['name']} image {i+1}/{len(pending)}",flush=True)
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def measure(args, model, folder):
    common.write_json(folder/'incomplete.json', dict(stage='measurement'))
    manifest = read_manifest(folder)
    expected = plan(args, model)
    for key in ('model_id','architecture','domain','images','pairs','seed','samples'):
        if manifest.get(key) != expected[key]:
            raise ValueError('Image generation design differs from measurement request: '+key)
    for key,wanted in [('steps',args.steps),('guidance_scale',args.guidance_scale),('height',args.height),('width',args.width)]:
        if manifest['generation'][key] != wanted:
            raise ValueError('Generation setting differs: '+key)
    path_args = SimpleNamespace(revision='local')
    identity = common.source_identity(dict(model_id=args.vae_model,lora_id=args.segmenter_weights),path_args)
    protocol = dict(version='paper_stage2_measurement_v1', encoders=identity, vae_size=args.vae_size,
        pixel_size=args.pixel_size, segment_size=args.segment_size, mask_size=args.mask_size,
        mask_margin=args.mask_margin, min_common_background=args.min_common_background,
        foreground_classes=['person'], yolo_conf=.25, yolo_iou=.7, yolo_max_det=300,
        dtype='float32', zero_detections='full_image_background', generation_digest=common.digest(manifest))
    write_protocol(folder/'measurement.json',protocol)
    encoded, backend = {}, None
    try:
        for row in manifest['images']:
            image = folder/row['path']
            if not image_valid(image): raise ValueError('Missing or unreadable generated image: '+str(image))
            fingerprint = common.digest(dict(protocol=protocol,image_sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
            saved = folder/'features'/f"{row['id']}.npz"
            record = None
            if saved.is_file():
                try:
                    with np.load(saved,allow_pickle=False) as data:
                        if str(data['fingerprint']) == fingerprint:
                            record = {k:data[k].copy() for k in ('latent','background','full_image_background')}
                except (OSError,ValueError,KeyError):
                    pass
            if record is None:
                if backend is None: backend = ImageFeatures(args)
                record = backend.encode(image)
                saved.parent.mkdir(parents=True,exist_ok=True)
                temp = saved.with_suffix('.tmp.npz')
                np.savez_compressed(temp, fingerprint=np.array(fingerprint), **record); temp.replace(saved)
            encoded[row['id']] = (image,record)
    finally:
        if backend is not None:
            import torch
            del backend
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    rows=[]
    for pair in manifest['pairs']:
        ap, a = encoded[pair['reference']]; bp, b = encoded[pair['perturbation']]
        am = a['background'] if a['background'].size else None
        bm = b['background'] if b['background'].size else None
        background,coverage,windows = background_ssim(ap,am,bp,bm,args.pixel_size,args.min_common_background)
        rows.append(dict(**pair,vae_sign_hash=sign_hash(a['latent'],b['latent']),
            full_ssim=full_ssim(ap,bp,args.pixel_size),background_ssim=background,
            common_background_fraction=coverage,ssim_windows=windows,
            full_image_background=bool(a['full_image_background'] or b['full_image_background'])))
    common.write_csv(folder/'pair_metrics.csv',rows)
    common.write_csv(folder/'model_metrics.csv',aggregate_pairs(rows,args.domain))
    (folder/'incomplete.json').unlink()


def finalize(args):
    root=common.local_path(args.output_dir)
    first=read_csv(common.local_path(args.stage1_csv))
    if args.metrics:
        statistics=read_csv(common.local_path(args.metrics))
    else:
        statistics=[]
        for row in first:
            folder=root/row['model_id']
            path=folder/'model_metrics.csv'
            if path.is_file() and not (folder/'incomplete.json').exists():
                statistics.extend(read_csv(path))
    result=decide(first,statistics,args.domain,args.variance_weight,args.threshold,args.denominator_epsilon)
    common.write_csv(root/'final_predictions.csv',result)
    common.write_json(root/'summary.json',summaries(result,'final_prediction'))
    common.write_json(root/'decision_protocol.json',dict(domain=args.domain,variance_weight=args.variance_weight,
        threshold=args.threshold,denominator_epsilon=args.denominator_epsilon,
        rule='(hash_mean + variance_weight * full_ssim_variance) / background_ssim_mean >= threshold',
        missing='retain_stage1',stage1_sha256=hashlib.sha256(common.local_path(args.stage1_csv).read_bytes()).hexdigest()))
    print('Final decisions:',root/'final_predictions.csv')


def main(argv=None):
    p=parser(); args=p.parse_args(argv)
    args.models_config=args.models_config or f'configs/{args.architecture}.json'
    args.output_dir=args.output_dir or f'outputs/{args.architecture}/stage2/{args.domain}'
    if args.steps is None: args.steps=30 if args.architecture in ('sd14','sd2') else 20
    if args.guidance_scale is None: args.guidance_scale={'sd14':7.5,'sd2':7.5,'sd35':4.5,'flux':3.5}[args.architecture]
    if args.samples < 1 or args.steps < 1 or min(args.vae_size,args.pixel_size,args.mask_size,args.segment_size) < 11 or not 0 <= args.min_common_background <= 1 or args.mask_margin < 0:
        p.error('Invalid measurement settings')
    if args.mode in ('decide','all') and not args.stage1_csv:
        p.error('--stage1_csv must point to actual Stage-I predictions')
    if args.mode=='decide':
        finalize(args); return
    config=load_config(args.models_config)
    if config['architecture']!=args.architecture: p.error('Configuration architecture mismatch')
    models=config['test_models']
    if args.only:
        missing=set(args.only)-{m['name'] for m in models}
        if missing: p.error('Unknown model identifiers: '+str(sorted(missing)))
        models=[m for m in models if m['name'] in args.only]
    if args.mode=='check':
        print(json.dumps(dict(domain=args.domain,models=len(models),images_per_model=len(plan(args,models[0])['images']),
            pairs_per_model=len(plan(args,models[0])['pairs']),generation_steps=args.steps,guidance_scale=args.guidance_scale,
            variance_weight=args.variance_weight,threshold=args.threshold),indent=2)); return
    first={(r['architecture'],r['model_id']):r for r in read_csv(common.local_path(args.stage1_csv))} if args.stage1_csv else None
    for model in models:
        if first is not None:
            row=first.get((args.architecture,model['name']))
            if row is None: raise ValueError('Missing Stage-I decision for '+model['name'])
            if int(row['stage1_prediction'])==0: continue
        folder=common.local_path(args.output_dir)/model['name']
        if args.mode in ('generate','all'): generate(args,model,folder)
        if args.mode in ('measure','all'): measure(args,model,folder)
    if args.mode=='all': finalize(args)


if __name__=='__main__':
    main()
