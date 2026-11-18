"""Train one detector per architecture, evaluate the model inventory, and run Stage II."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from detection import common
from detection.settings import STEPS, load_config, parse_args as stage1_args, protocol
from detection.extraction import load_anchors
from detection.evaluation import read_csv, summaries


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def call(script, args):
    subprocess.run([sys.executable,'-B',str(common.ROOT/script),*map(str,args)],cwd=common.ROOT,check=True)


def complete_features(path, expected):
    if not path.is_file() or not path.with_suffix('.meta.json').is_file(): return False
    try:
        rows,_ = common.read_features(path)
        return len(rows)==expected
    except (OSError,ValueError,KeyError,json.JSONDecodeError):
        return False


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--architecture',choices=['sd14','sd2','sd35','flux'],default='sd2')
    p.add_argument('--models_config',default=None)
    p.add_argument('--output',default=None)
    p.add_argument('--stages',choices=['stage1','stage2','all'],default='all')
    p.add_argument('--only',nargs='+')
    p.add_argument('--device',default='cuda')
    p.add_argument('--step_count',type=int,choices=list(STEPS),default=21)
    p.add_argument('--anchors',type=int,default=7)
    p.add_argument('--anchor_prompts',default='data/anchors.txt')
    p.add_argument('--clip_pretrained',default='models/clip-vit-large-patch14')
    p.add_argument('--vae_model',default='models/sd2/base/vae')
    p.add_argument('--segmenter_weights',default='models/yolo26x-seg.pt')
    p.add_argument('--domain',default='gender',choices=['gender','age','race','religion','brand','geo_cultural'])
    p.add_argument('--check',action='store_true',help='Print the plan without loading models or writing outputs')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    config=load_config(args.models_config or f'configs/{args.architecture}.json')
    if config['architecture']!=args.architecture: raise ValueError('Configuration architecture mismatch')
    models=config['test_models']
    if args.only:
        if set(args.only)-{m['name'] for m in models}: raise ValueError('Unknown model selection')
        models=[m for m in models if m['name'] in args.only]
    root=common.local_path(args.output or f'outputs/{args.architecture}')
    train=root/'stage1'; checkpoint=train/'detector.pt'
    config_path=common.local_path(args.models_config or f'configs/{args.architecture}.json')
    shared=['--architecture',args.architecture,'--models_config',config_path,
            '--device',args.device,'--capture_steps',','.join(map(str,STEPS[args.step_count])),
            '--anchor_prompts',args.anchor_prompts,'--max_anchors',args.anchors,
            '--clip_pretrained',args.clip_pretrained]
    options=stage1_args(list(map(str,shared)))
    anchors=load_anchors(options)
    if args.check:
        print(json.dumps(dict(architecture=args.architecture,anchors=anchors,steps=STEPS[args.step_count],
            numerical_shape=[args.step_count,10*len(anchors)+5],train_models=len(config['train_models']),
            test_models=len(models),output=str(root),stages=args.stages,domain=args.domain),indent=2));return
    if args.stages in ('stage1','all'):
        plan=dict(architecture=args.architecture,protocol=protocol(options,anchors),configuration=config,
            train_prompts=fingerprint(common.local_path(options.train_prompts)),
            test_prompts=fingerprint(common.local_path(options.test_prompts)),
            sources=[common.source_identity(m,options) for m in config['train_models']+config['test_models']],
            image_encoder=common.source_identity(dict(model_id=options.clip_pretrained),options))
        manifest=train/'run.json'
        if manifest.is_file() and json.loads(manifest.read_text(encoding='utf-8'))!=json.loads(json.dumps(plan)):
            raise ValueError('Inputs or protocol changed. Use a new --output directory.')
        common.write_json(manifest,plan)
        train_features=train/'features_train.jsonl'
        if not complete_features(train_features,100*len(config['train_models'])):
            call('stage1.py',shared+['--mode','extract','--dataset','train','--output_dir',train])
        training_stamp=dict(features=fingerprint(train_features),defaults={k:getattr(options,k) for k in
            ('seed','epochs','patience','batch_size','learning_rate','weight_decay','hidden_dim','attention_heads',
             'channel_layers','transformer_layers','dropout','normalization_clip','validation_models','threshold')})
        marker=train/'trained.json'
        if not (checkpoint.is_file() and checkpoint.with_suffix('.json').is_file() and marker.is_file()
                and json.loads(marker.read_text(encoding='utf-8'))==training_stamp):
            call('stage1.py',shared+['--mode','train','--output_dir',train,'--checkpoint',checkpoint])
            common.write_json(marker,training_stamp)
        metadata=json.loads(checkpoint.with_suffix('.json').read_text(encoding='utf-8'))
        checkpoint_hash=fingerprint(checkpoint)
        statuses=[]
        for model in models:
            folder=train/'batch_test'/model['name']; features=folder/'features_test.jsonl'
            try:
                if not complete_features(features,100):
                    call('stage1.py',shared+['--mode','extract','--dataset','test','--output_dir',folder,
                        '--model_name',model['name'],'--target_model',model['model_id'],'--target_lora',model.get('lora_id','')])
                stamp=dict(features=fingerprint(features),checkpoint=checkpoint_hash)
                finished=folder/'complete.json'
                if not (finished.is_file() and (folder/'stage1.csv').is_file() and (folder/'prompt_predictions.csv').is_file()
                        and json.loads(finished.read_text(encoding='utf-8'))==stamp):
                    call('stage1.py',shared+['--mode','predict','--output_dir',folder,'--checkpoint',checkpoint])
                    common.write_json(finished,stamp)
                else:
                    print('[skip] completed model:',model['name'],flush=True)
                statuses.append(dict(model_id=model['name'],status='ok',error=''))
            except (OSError,ValueError,subprocess.CalledProcessError) as error:
                statuses.append(dict(model_id=model['name'],status='error',error=str(error)))
                print('[error]',model['name'],error,flush=True)
            common.write_csv(train/'status.csv',statuses)
        # Include earlier completed models when extending a subset run.
        combined=[]
        for model in config['test_models']:
            folder=train/'batch_test'/model['name']; path=folder/'stage1.csv'; marker=folder/'complete.json'
            if not path.is_file() or not marker.is_file(): continue
            if json.loads(marker.read_text(encoding='utf-8')).get('checkpoint')!=checkpoint_hash: continue
            row=read_csv(path)
            if len(row)!=1 or row[0]['model_id']!=model['name']: raise ValueError('Unexpected model result: '+str(path))
            row=row[0]; row['true_label']=model['label']
            identity=common.digest(common.source_identity(model,options))
            row['train_validation_overlap']='source_model' if identity in metadata['training_sources'] else 'none'
            combined.append(row)
        if combined:
            common.write_csv(train/'stage1.csv',combined)
            common.write_json(train/'summary.json',dict(expected_models=len(config['test_models']),
                completed_models=len(combined),**summaries(combined)))
        if any(s['status']!='ok' for s in statuses):
            raise SystemExit('Some Stage-I models failed; completed outputs are preserved for resume.')
    if args.stages in ('stage2','all'):
        first=train/'stage1.csv'
        if not first.is_file(): raise FileNotFoundError('Complete Stage I first: '+str(first))
        decisions={r['model_id']:r for r in read_csv(first)}
        second=root/'stage2'/args.domain
        shared2=['--architecture',args.architecture,'--models_config',config_path,'--device',args.device,
            '--domain',args.domain,'--output_dir',second,'--stage1_csv',first,
            '--vae_model',args.vae_model,'--segmenter_weights',args.segmenter_weights]
        errors=[]
        for model in models:
            if model['name'] not in decisions: raise ValueError('Missing Stage-I decision for '+model['name'])
            if int(decisions[model['name']]['stage1_prediction'])==0: continue
            common.write_json(second/model['name']/'incomplete.json',dict(stage='pending'))
            try:
                for mode in ('generate','measure'):
                    call('stage2.py',shared2+['--mode',mode,'--only',model['name']])
            except (OSError,subprocess.CalledProcessError):
                errors.append(model['name'])
        call('stage2.py',shared2+['--mode','decide'])
        if errors: raise SystemExit('Stage-II failures retained as Stage-I decisions: '+', '.join(errors))


if __name__=='__main__':
    main()
