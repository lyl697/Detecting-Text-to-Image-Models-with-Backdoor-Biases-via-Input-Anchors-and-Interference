"""Ablation studies: anchor count, capture density, feature groups, and prompt budget."""
import argparse
import copy
import json
import random
import statistics
from pathlib import Path
from detection import common, detector
from detection.settings import STEPS, load_config, parse_args as stage1_args
from detection.protocol import feature_metadata
from detection.evaluation import read_csv, aggregate_rows, metrics, summaries
from run_experiment import call

GROUPS = {
    'block': ('lfd_mean','lfd_std','lfd_delta_mean'),
    'leaf': ('leaf_all_mean',),
    'image': ('image_clip_sim','image_lpips','image_mse','image_ssim','image_hog_pearson','image_hog_dtw'),
    'trajectory': (),
}


def subset_features(source, destination, variant):
    rows,_=detector.read_features(source)
    old=rows[0]['layout']; layout=copy.deepcopy(old)
    if variant=='without_trajectory': layout['trajectory_features']=[]
    elif variant!='full':
        removed=GROUPS[variant.removeprefix('without_')]
        layout['anchor_features']=[x for x in layout['anchor_features'] if x not in removed]
    meta=feature_metadata(source,rows[0]['signature'])
    if meta is None: raise ValueError('Missing feature protocol sidecar')
    meta=copy.deepcopy(meta); meta['protocol']['feature_ablation']=variant
    names=detector.layout_names(layout); old_names=rows[0]['feature_names']
    positions=[old_names.index(n) for n in names]
    signature=common.digest(meta)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        for row in rows:
            row.update(layout=layout,feature_names=names,signature=signature,
                       values=[[step[i] for i in positions] for step in row['values']])
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    temp.replace(destination); common.write_json(destination.with_suffix('.meta.json'),meta)


def feature_ablation(args):
    source=common.local_path(args.source)
    checkpoint=json.loads((source/'detector.json').read_text(encoding='utf-8'))
    config=load_config(args.models_config or f'configs/{args.architecture}.json')
    output=common.local_path(args.output or f'outputs/experiments/{args.architecture}/features')
    variants=['full']+['without_'+g for g in GROUPS]
    results=[]
    for variant in variants:
        folder=output/variant
        subset_features(source/'features_train.jsonl',folder/'features_train.jsonl',variant)
        options=stage1_args(['--architecture',args.architecture,'--output_dir',str(folder),'--device',args.device])
        original=checkpoint['training_args']
        for key in ('seed','epochs','patience','batch_size','learning_rate','weight_decay','val_fraction','threshold',
                    'hidden_dim','attention_heads','channel_layers','transformer_layers','dropout','normalization_clip'):
            setattr(options,key,original[key])
        options.validation_models=','.join(checkpoint['validation_models'])
        detector.train(options)
        fitted=json.loads((folder/'detector.json').read_text(encoding='utf-8'))
        if fitted['train_models']!=checkpoint['train_models'] or fitted['validation_models']!=checkpoint['validation_models']:
            raise ValueError('Feature ablation changed the source-model split')
        rows=[]; missing=[]
        for model in config['test_models']:
            path=source/'batch_test'/model['name']/'features_test.jsonl'
            if not path.is_file(): missing.append(model['name']); continue
            out=folder/'batch_test'/model['name']
            subset_features(path,out/'features_test.jsonl',variant)
            options.output_dir=str(out)
            detector.predict(options)
            result=read_csv(out/'stage1.csv')[0];result['true_label']=model['label']
            with path.open(encoding='utf-8') as stream: first=json.loads(next(stream))
            result['train_validation_overlap']='source_model' if first['source_identity'] in checkpoint['training_sources'] else 'none'
            rows.append(result)
        if rows: common.write_csv(folder/'stage1.csv',rows)
        summary=dict(variant=variant,missing_models=missing,**summaries(rows))
        common.write_json(folder/'summary.json',summary)
        results.append(dict(variant=variant,completed_models=len(rows),**metrics(rows)))
    common.write_csv(output/'feature_ablation.csv',results)


def prompt_budget(args):
    source=common.local_path(args.source)
    model_rows=read_csv(source/'stage1.csv')
    groups={}; texts=None
    for row in model_rows:
        records=read_csv(source/'batch_test'/row['model_id']/'prompt_predictions.csv')
        records.sort(key=lambda r:int(r['prompt_index']))
        if len(records)!=100 or [int(r['prompt_index']) for r in records]!=list(range(100)):
            raise ValueError('Prompt-budget experiments require the complete ordered 100-prompt pool')
        current=[r['original_prompt'] for r in records]
        if texts is not None and current!=texts: raise ValueError('Prompt sets differ across models')
        texts=current
        for r in records: r['true_label']=row['true_label']
        groups[row['model_id']]=records
    counts=[int(s) for s in args.counts.split(',')] if args.counts else [10,20,50,100]
    if any(not 1<=n<=100 for n in counts) or len(set(counts))!=len(counts): raise ValueError('Invalid prompt budgets')
    rng=random.Random(args.seed); output=common.local_path(args.output or str(source/'prompt_budget'))
    runs=[]; selections=[]
    for repeat in range(args.repeats):
        order=list(range(100));rng.shuffle(order)
        selections.append(dict(repeat=repeat,indices=order))
        for count in counts:
            if count==100 and repeat>0: continue
            selected=sorted(order[:count]); predictions=[]
            for model in model_rows:
                result=aggregate_rows([groups[model['model_id']][i] for i in selected],float(model['stage1_threshold']))[0]
                result['train_validation_overlap']=model.get('train_validation_overlap','unknown')
                predictions.append(result)
            for cohort, values in summaries(predictions).items():
                runs.append(dict(count=count,repeat=repeat,cohort=cohort,**values))
    result=[]
    for count in counts:
        for cohort in ('all_labeled','excluding_train_validation_overlap'):
            acc=[r['accuracy'] for r in runs if r['count']==count and r['cohort']==cohort and r['accuracy'] is not None]
            result.append(dict(count=count,cohort=cohort,repeats=len(acc),accuracy_mean=statistics.mean(acc) if acc else None,
                               accuracy_std=statistics.pstdev(acc) if acc else None))
    common.write_csv(output/'summary.csv',result);common.write_csv(output/'repeats.csv',runs)
    common.write_json(output/'selection.json',dict(seed=args.seed,repeats=args.repeats,selection='global random nested prefixes',
        models=[r['model_id'] for r in model_rows],prompts=texts,selections=selections))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['anchors','steps','features','prompts'],required=True)
    p.add_argument('--architecture',choices=['sd14','sd2','sd35','flux'],default='sd2')
    p.add_argument('--models_config',default=None)
    p.add_argument('--source',help='Stage-I directory containing detector.json and batch_test/')
    p.add_argument('--output',default=None)
    p.add_argument('--counts',default=None)
    p.add_argument('--device',default='cuda')
    p.add_argument('--repeats',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--check',action='store_true')
    args=p.parse_args(argv)
    if args.repeats<1:p.error('repeats must be positive')
    if args.mode in ('features','prompts'):
        if not args.source:p.error('--source is required')
        if args.check:
            print('Source:',common.local_path(args.source));return
        (feature_ablation if args.mode=='features' else prompt_budget)(args);return
    defaults={'anchors':'7,14,21,28,35','steps':'6,11,21,50'}
    counts=[int(s) for s in (args.counts or defaults[args.mode]).split(',')]
    allowed=set([7,14,21,28,35] if args.mode=='anchors' else STEPS)
    if not counts or not set(counts)<=allowed or len(set(counts))!=len(counts):p.error('Invalid counts')
    root=common.local_path(args.output or f'outputs/experiments/{args.architecture}/{args.mode}')
    results=[]
    for count in counts:
        folder=root/f'{args.mode}_{count:02d}'
        command=['--architecture',args.architecture,'--device',args.device,'--stages','stage1','--output',folder,
                 '--anchors',count if args.mode=='anchors' else 7,'--step_count',6 if args.mode=='anchors' else count]
        if args.models_config:command+=['--models_config',args.models_config]
        if args.check:command+=['--check']
        call('run_experiment.py',command)
        if not args.check:
            summary=json.loads((folder/'stage1/summary.json').read_text(encoding='utf-8'))
            results.append(dict(count=count,**summary['all_labeled']))
    if results:common.write_csv(root/'summary.csv',results)


if __name__=='__main__':
    main()
