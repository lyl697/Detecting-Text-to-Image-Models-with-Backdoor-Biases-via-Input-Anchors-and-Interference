"""Stage-I command-line entry point."""
import json
from detection.settings import parse_args, protocol, model_specs, feature_layout
from detection.extraction import load_anchors, feature_names, extract, train, predict


def main(argv=None):
    args = parse_args(argv)
    if args.mode == 'check':
        anchors = load_anchors(args)
        config = protocol(args, anchors)
        print(json.dumps(dict(architecture=args.architecture, anchors=anchors,
            capture_steps=config['capture']['steps'], features_per_step=len(feature_names(config)),
            layout=feature_layout(config), training_models=model_specs(args, 'train'),
            defaults=vars(args)), indent=2))
    elif args.mode == 'extract':
        extract(args, args.dataset)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
    else:
        extract(args, 'train')
        train(args)
        extract(args, 'test')
        predict(args)


if __name__ == '__main__':
    main()
