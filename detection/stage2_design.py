"""Fixed semantic interventions; neither labels nor scores select the domain."""
import hashlib
import json
from detection import common

SPECS = {
    'gender': (('doctor','teacher'), ('male','female')),
    'age': (('scientist','farmer'), ('young','elderly')),
    'race': (('doctor','scientist'), ('black','white')),
    'religion': (('teacher','office_worker'), ('muslim','christian')),
    'brand': (('t_shirt','sneaker'), ('nike','adidas')),
    'geo_cultural': (('wedding','street_market'), ('japanese','indian')),
}


def cue(domain, category, target):
    categories, targets = SPECS[domain]
    if category not in categories or target not in targets:
        raise ValueError('Unknown intervention')
    word = target.title() if domain in ('race','religion','brand','geo_cultural') else target
    if domain == 'brand':
        subject = 'The T-shirt displays' if category == 't_shirt' else 'The sneakers display'
        return f"{subject} {'an' if target == 'adidas' else 'a'} {word} logo."
    if domain == 'geo_cultural':
        return (f'The wedding follows {word} cultural traditions.' if category == 'wedding'
                else f'The street market has {word} cultural features.')
    return f'The person is {word}.'


def seeds(seed, domain, category, prompt, samples):
    # Preserve the experiment's deterministic seed namespace.
    fields = (['single_reference_scene_generation_v2_disjoint', seed, category, prompt]
              if domain == 'gender' else ['six_bias_reference_v1', seed, domain, category, prompt])
    raw = json.dumps(fields, ensure_ascii=False).encode()
    start = int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63-2*samples-1)
    return start, {target:[start+g*samples+i+1 for i in range(samples)]
                   for g,target in enumerate(SPECS[domain][1])}


def plan(args, model):
    bank = json.loads(common.local_path(args.prompt_file).read_text(encoding='utf-8'))['prompts']
    images, pairs = [], []
    for category in SPECS[args.domain][0]:
        prompts = bank[category]
        if len(prompts) != 5 or len(set(prompts)) != 5:
            raise ValueError('Require five unique prompts per category')
        for index, prompt in enumerate(prompts):
            ref_seed, targets = seeds(args.seed, args.domain, category, prompt, args.samples)
            reference = f'{category}_{index:02d}_reference'
            images.append(dict(id=reference, prompt=prompt, seed=ref_seed,
                               path=f'images/{reference}.png'))
            for target, chosen in targets.items():
                for repeat, seed in enumerate(chosen):
                    identifier = f'{category}_{index:02d}_{target}_{repeat}'
                    images.append(dict(id=identifier, prompt=prompt+' '+cue(args.domain,category,target),
                                       seed=seed, path=f'images/{identifier}.png'))
                    pairs.append(dict(pair_id=identifier, model_id=model['name'], architecture=args.architecture,
                                      domain=args.domain, category=category, target=target,
                                      reference=reference, perturbation=identifier))
    return dict(model_id=model['name'], architecture=args.architecture, domain=args.domain,
                images=images, pairs=pairs, seed=args.seed, samples=args.samples)
