"""Editable defaults for Stage-I feature extraction and classification."""
import argparse
import json
import re
from detection import common
from detection.feature_math import TRACE_NAMES
from detection.relations import SCHEMA, RAW_SCHEMA

STEPS = {
    6: [0, 10, 20, 30, 40, 49],
    11: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49],
    21: [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45, 47, 49],
    50: list(range(50)),
}
IMAGE_METRICS = ('clip_sim', 'lpips', 'mse', 'ssim', 'hog_pearson', 'hog_dtw')


def parser():
    p = argparse.ArgumentParser(description='Extract features, train, or evaluate the Stage-I detector.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode', choices=['check', 'extract', 'train', 'predict', 'all'], default='check')
    p.add_argument('--dataset', choices=['train', 'test'], default='train')
    p.add_argument('--architecture', choices=['sd14', 'sd2', 'sd35', 'flux'], default='sd2')
    p.add_argument('--models_config', default=None, help='Default: configs/<architecture>.json')
    p.add_argument('--target_model', default=None)
    p.add_argument('--target_lora', default='')
    p.add_argument('--model_name', default='suspect')
    p.add_argument('--train_prompts', default='data/train.txt')
    p.add_argument('--test_prompts', default='data/test.txt')
    p.add_argument('--max_prompts', type=int, default=100)
    p.add_argument('--anchor_prompts', default='data/anchors.txt')
    p.add_argument('--max_anchors', type=int, default=7)
    p.add_argument('--num_inference_steps', type=int, default=50)
    p.add_argument('--capture_steps', default=','.join(map(str, STEPS[21])))
    p.add_argument('--height', type=int, default=768)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--guidance_scale', type=float, default=None, help='7.5; FLUX: 4.5')
    p.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default=None,
                   help='float16; FLUX: bfloat16; CPU: float32')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num_images', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--local_files_only', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--revision', default='main')
    p.add_argument('--clip_pretrained', default='models/clip-vit-large-patch14')
    p.add_argument('--output_dir', default=None, help='Default: outputs/<architecture>/stage1')
    p.add_argument('--checkpoint', default=None, help='Default: <output_dir>/detector.pt')
    p.add_argument('--cache_dir', default='cache/stage1')
    p.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--validation_models', default='backdoor_1,clean_3')
    p.add_argument('--val_fraction', type=float, default=.25)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=.001)
    p.add_argument('--weight_decay', type=float, default=.0001)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--attention_heads', type=int, default=4)
    p.add_argument('--channel_layers', type=int, default=1)
    p.add_argument('--transformer_layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--normalization_clip', type=float, default=10.)
    p.add_argument('--threshold', type=float, default=.5)
    return p


def parse_args(argv=None):
    args = parser().parse_args(argv)
    args.models_config = args.models_config or f'configs/{args.architecture}.json'
    args.target_model = args.target_model or f'models/{args.architecture}/base'
    args.output_dir = args.output_dir or f'outputs/{args.architecture}/stage1'
    args.checkpoint = args.checkpoint or str(common.local_path(args.output_dir) / 'detector.pt')
    args.guidance_scale = args.guidance_scale if args.guidance_scale is not None else (4.5 if args.architecture == 'flux' else 7.5)
    args.dtype = args.dtype or ('float32' if args.device == 'cpu' else 'bfloat16' if args.architecture == 'flux' else 'float16')
    args.denoiser_component, args.block_output_index = 'auto', -1
    return args


def load_config(path):
    config = json.loads(common.local_path(path).read_text(encoding='utf-8-sig'))
    if config.get('architecture') not in ('sd14', 'sd2', 'sd35', 'flux'):
        raise ValueError('Unknown architecture in model configuration')
    for section in ('train_models', 'test_models'):
        rows = config[section]
        if not rows or len({r['name'] for r in rows}) != len(rows):
            raise ValueError('Empty or duplicated model identifiers: ' + section)
        identities = set()
        for r in rows:
            if not re.fullmatch(r'[A-Za-z0-9_-]+', r['name']) or r.get('label') not in (0, 1):
                raise ValueError('Models require a safe unique name and binary label')
            if r.get('architecture', config['architecture']) != config['architecture']:
                raise ValueError('Do not mix architectures in a detector')
            identity = (str(r['model_id']), str(r.get('lora_id') or ''))
            if identity in identities:
                raise ValueError('Duplicate checkpoint/adapter within ' + section)
            identities.add(identity)
    return config


def model_specs(args, dataset):
    if dataset == 'test':
        return [dict(name=args.model_name, model_id=args.target_model, lora_id=args.target_lora,
                     architecture=args.architecture, label=None)]
    config = load_config(args.models_config)
    if config['architecture'] != args.architecture:
        raise ValueError('Configuration architecture differs from --architecture')
    return config['train_models']


def protocol(args, anchors):
    steps = [int(s) for s in args.capture_steps.split(',')]
    if args.num_inference_steps != 50 or len(steps) < 2 or steps[0] != 0 or steps[-1] > 49 or any(b <= a for a,b in zip(steps,steps[1:])):
        raise ValueError('Use 50 inference steps and increasing callback indices within 0..49, starting at 0')
    if args.max_prompts < 0 or args.num_images < 1 or min(args.height,args.width) < 32:
        raise ValueError('Invalid generation or prompt settings')
    capture = dict(schema=RAW_SCHEMA, steps=steps, step_grid='fixed_indices_v1',
                   seed_rule='same seed+replica for every anchor and probe',
                   **{key:getattr(args,key) for key in ('num_inference_steps','guidance_scale','height','width',
                       'num_images','seed','dtype','denoiser_component','block_output_index')})
    return dict(schema=SCHEMA, capture=capture, anchors=anchors, pools=['mean','std'], groups=['all'],
                metrics=list(IMAGE_METRICS), centering='none', temporal=True, min_fraction=.5, eps=1e-12,
                block_statistics='mean,std,delta_mean', require_all_blocks_valid=True)


def native_capture_grid(pipe, capture):
    timesteps = getattr(pipe.scheduler, 'timesteps', None)
    if timesteps is None or capture['steps'][-1] >= len(timesteps):
        raise ValueError('Requested callback indices exceed the initialized scheduler grid')
    return len(timesteps), {step:step for step in capture['steps']}


def feature_layout(config):
    return dict(anchors=config['anchors'], anchor_features=['lfd_mean','lfd_std','lfd_delta_mean',
                'leaf_all_mean'] + ['image_'+s for s in config['metrics']], trajectory_features=list(TRACE_NAMES))
