"""Load the fixed pretrained CLIP vision tower and its projection."""
import json
from detection import common


def load_clip_encoder(args, device, root):
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    source = common.resolve_model(args.clip_pretrained)
    model, info = CLIPVisionModelWithProjection.from_pretrained(
        source, local_files_only=args.local_files_only, output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('Incomplete/incompatible pretrained CLIP vision weights: ' + str(info))
    processor = CLIPImageProcessor.from_pretrained(source, local_files_only=args.local_files_only)
    identity = dict(backend='transformers', model=type(model).__name__,
                    preprocess=json.loads(json.dumps(processor.to_dict())))
    return 'transformers', model.to(device).requires_grad_(False).eval(), processor, identity
