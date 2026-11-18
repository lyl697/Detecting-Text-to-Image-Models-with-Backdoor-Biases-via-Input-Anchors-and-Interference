import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from PIL import Image
from detection import common, detector
from detection.settings import parse_args, protocol, feature_layout, load_config
from detection.extraction import load_anchors, fuse_prompt, write_npz, relation_file
from detection.relations import group_relations, lfd_relation_features, SCHEMA
from detection.feature_math import TRACE_NAMES, latent_trajectory, grouped_split, sample_weights
from detection.stage2_design import plan, seeds
from detection.stage2_features import sign_hash, full_ssim, background_ssim
from detection.stage2_decision import aggregate_pairs, composite_score, decide
from detection.evaluation import aggregate_rows
from experiments import subset_features
from stage2 import write_protocol, finalize
from detection.evaluation import read_csv


class Features(unittest.TestCase):
    def test_paper_defaults_for_all_architectures(self):
        for architecture in ('sd14','sd2','sd35','flux'):
            args=parse_args(['--architecture',architecture]); config=protocol(args,load_anchors(args))
            layout=feature_layout(config)
            self.assertEqual(len(config['capture']['steps']),21)
            self.assertEqual(len(layout['anchors'])*len(layout['anchor_features'])+5,75)
            self.assertEqual(layout['anchor_features'][:3],['lfd_mean','lfd_std','lfd_delta_mean'])
            self.assertFalse(any('valid_fraction' in f for f in layout['anchor_features']))
            self.assertEqual(layout['trajectory_features'],list(TRACE_NAMES))

    def test_model_inventories_and_source_split(self):
        for architecture,overlap in [('sd14',5),('sd2',5),('sd35',5),('flux',6)]:
            config=load_config(f'configs/{architecture}.json')
            self.assertEqual(len(config['test_models']),40)
            self.assertEqual(sum(r['label'] for r in config['test_models']),20)
            identity=lambda r:(r['model_id'],r['lora_id'])
            sources={identity(r) for r in config['train_models']}
            self.assertEqual(sum(identity(r) in sources for r in config['test_models']),overlap)
            rows=[dict(model_id=r['name'],label=r['label']) for r in config['train_models'] for _ in range(100)]
            ti,vi=grouped_split(rows,.25,42,['backdoor_1','clean_3'])
            self.assertEqual((len(ti),len(vi)),(400,200))
            weights=sample_weights([rows[i] for i in ti])
            self.assertAlmostEqual(float(weights.sum()),1,places=6)

    def test_three_block_statistics(self):
        valid=np.ones(3,bool)
        probe={('block','a'):(np.array([[1,0],[1,0],[1,0]]),valid),
               ('block','b'):(np.array([[1,0],[1,0],[1,0]]),valid)}
        anchor={('block','a'):(np.array([[1,0],[0,1],[-1,0]]),valid),
                ('block','b'):(np.array([[0,1],[1,0],[0,1]]),valid)}
        pooled,coverage=group_relations(probe,[anchor,anchor],'block',3)
        np.testing.assert_allclose(coverage,1)
        values=lfd_relation_features(pooled,coverage).reshape(3,3,2)
        np.testing.assert_allclose(values[:,0,0],[.5,.5,-.5])
        np.testing.assert_allclose(values[:,1,0],[.5,.5,.5])
        np.testing.assert_allclose(values[:,2,0],[0,0,-1])

    def test_anchor_major_fusion(self):
        args=parse_args(['--max_anchors','2','--capture_steps','0,49','--num_images','1'])
        config=protocol(args,load_anchors(args)); steps,count=2,2
        stats=np.arange(steps*3*count,dtype=np.float32).reshape(1,steps,3*count)
        trace=np.ones((1,steps,5),np.float32)*100
        leaves=np.ones((1,steps,1,count),np.float32)*20
        scorer=SimpleNamespace(compare=lambda *a:[1,2,3,4,5,6])
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            write_npz(relation_file(folder,'probe'),lfd=stats,trace=trace,module_mean=leaves)
            result=fuse_prompt(folder,folder,'probe',scorer,args,config)
        self.assertEqual(result.shape,(2,25))
        np.testing.assert_equal(result[0,:10],[0,2,4,20,1,2,3,4,5,6])
        np.testing.assert_equal(result[0,10:20],[1,3,5,20,1,2,3,4,5,6])
        np.testing.assert_equal(result[:,-5:],100)

    def test_normalization_train_only_and_masks(self):
        layout=dict(anchors=['a','b'],anchor_features=['f'],trajectory_features=['t'])
        x=np.array([[[1,999,3,1,0],[3,5,5,1,1]]],dtype=np.float32)
        mean,std=detector.fit_normalizer(x,layout)
        np.testing.assert_allclose(mean,[2,5,4])
        normalized=detector.normalize(x,mean,std,10,layout)
        self.assertEqual(normalized[0,0,1],0)
        np.testing.assert_array_equal(normalized[...,3:],x[...,3:])

    def test_latent_first_state(self):
        features=latent_trajectory([np.ones(4),np.ones(4)*2])
        np.testing.assert_allclose(features[0,[0,1,3,4]],[0,1,0,0])
        self.assertEqual(features.shape,(2,5))

    def test_ablation_is_actual_column_removal(self):
        args=parse_args(['--max_anchors','2','--capture_steps','0,49'])
        config=protocol(args,load_anchors(args));layout=feature_layout(config)
        names=detector.layout_names(layout); meta=dict(protocol=config,encoder={})
        row=dict(schema=SCHEMA,signature=common.digest(meta),layout=layout,feature_names=names,
                 model_id='m',architecture='sd2',label=0,source_identity='source',prompt_index=0,
                 original_prompt='probe',anchor_valid=[[True,True],[True,True]],
                 values=np.arange(2*len(names)).reshape(2,-1).tolist())
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.jsonl';source.write_text(json.dumps(row)+'\n',encoding='utf-8')
            common.write_json(source.with_suffix('.meta.json'),meta)
            for variant,width in [('full',25),('without_block',19),('without_leaf',23),('without_image',13),('without_trajectory',20)]:
                dest=Path(tmp)/(variant+'.jsonl');subset_features(source,dest,variant)
                rows,packed=detector.read_features(dest)
                self.assertEqual(len(rows[0]['feature_names']),width)
                self.assertEqual(packed.shape,(1,2,width+2))
                self.assertNotEqual(rows[0]['signature'],row['signature'])


class Validation(unittest.TestCase):
    @staticmethod
    def first(probability=.8, model='m'):
        return dict(model_id=model,architecture='sd2',stage1_score=probability,stage1_threshold=.5,
                    stage1_prediction=int(probability>.5),true_label=0)

    @staticmethod
    def metric(h=.7,var=.01,bg=.2,model='m'):
        return dict(model_id=model,architecture='sd2',domain='gender',hash_mean=h,
                    full_ssim_variance=var,background_ssim_mean=bg,hash_n=80,full_n=80,background_n=80)

    def test_threshold_inclusive_and_formula(self):
        row=self.metric(h=.5,var=.05,bg=.25)
        self.assertEqual(composite_score(row),(4.,'ok'))
        result=decide([self.first()],[row])[0]
        self.assertEqual(result['final_prediction'],0)
        self.assertEqual(result['stage2_action'],'clear')

    def test_only_stage1_positive_can_change(self):
        result=decide([self.first(.2)],[self.metric(bg=.9)])[0]
        self.assertEqual(result['final_prediction'],0)
        self.assertEqual(result['stage2_status'],'not_required')

    def test_missing_and_invalid_denominator_retain(self):
        for metric in [{},self.metric(bg=0),self.metric(bg=-.2),self.metric(bg=1e-10),self.metric(h=float('nan'))]:
            rows=[] if not metric else [metric]
            result=decide([self.first()],rows)[0]
            self.assertEqual(result['final_prediction'],1)
            self.assertEqual(result['stage2_action'],'retain_unavailable')

    def test_positive_below_threshold_retained(self):
        result=decide([self.first()],[self.metric(bg=.9)])[0]
        self.assertEqual(result['final_prediction'],1)
        self.assertEqual(result['stage2_action'],'retain')

    def test_labels_do_not_change_decisions(self):
        first=self.first();a=decide([first],[self.metric()])[0]
        first['true_label']=1;b=decide([first],[self.metric()])[0]
        self.assertEqual(a['stage2_score'],b['stage2_score'])
        self.assertEqual(a['final_prediction'],b['final_prediction'])

    def test_reject_inconsistent_stage1(self):
        row=self.first(.5);row['stage1_prediction']=1
        with self.assertRaises(ValueError):decide([row],[])

    def test_duplicate_models_rejected(self):
        with self.assertRaises(ValueError):decide([self.first(),self.first()],[])
        with self.assertRaises(ValueError):decide([self.first()],[self.metric(),self.metric()])

    def test_fixed_domain(self):
        metric=self.metric();metric['domain']='age'
        self.assertEqual(decide([self.first()],[metric])[0]['stage2_action'],'retain_unavailable')

    def test_metric_specific_pair_pooling(self):
        rows=[dict(model_id='m',architecture='sd2',domain='gender',pair_id=str(i),
                   vae_sign_hash=h,full_ssim=f,background_ssim=b)
              for i,(h,f,b) in enumerate([(.6,.2,.1),(.8,.6,None),(None,None,.3)])]
        result=aggregate_pairs(rows)[0]
        self.assertAlmostEqual(result['hash_mean'],.7)
        self.assertAlmostEqual(result['full_ssim_variance'],.04)
        self.assertAlmostEqual(result['background_ssim_mean'],.2)
        self.assertEqual([result[k] for k in ['hash_n','full_n','background_n']],[2,2,2])
        with self.assertRaises(ValueError):aggregate_pairs(rows+[rows[0]])

    def test_seed_grid_and_image_counts(self):
        args=SimpleNamespace(domain='gender',samples=4,seed=42,prompt_file='data/stage2_prompts.json',architecture='sd2')
        design=plan(args,{'name':'m'})
        self.assertEqual((len(design['images']),len(design['pairs'])),(90,80))
        seed,targets=seeds(42,'gender','doctor','A doctor.',4)
        self.assertEqual(len({seed,*targets['male'],*targets['female']}),9)
        self.assertEqual((seed,targets),seeds(42,'gender','doctor','A doctor.',4))

    def test_pixel_and_sign_measurements(self):
        self.assertEqual(sign_hash([0,-1,1],[1,-2,-1]),2/3)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.png'
            rng=np.random.default_rng(42)
            Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(path)
            self.assertAlmostEqual(full_ssim(path,path,32),1.)
            score,coverage,windows=background_ssim(path,np.ones((32,32),bool),path,np.ones((32,32),bool),32)
            self.assertAlmostEqual(score,1.);self.assertEqual(coverage,1.);self.assertEqual(windows,22*22)
            score,_,_=background_ssim(path,np.zeros((32,32),bool),path,np.ones((32,32),bool),32)
            self.assertIsNone(score)

    def test_resume_accepts_serialized_tuple_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'protocol.json'
            write_protocol(path,dict(files=[('file',12,34)]))
            write_protocol(path,dict(files=[('file',12,34)]))
            with self.assertRaises(ValueError):write_protocol(path,dict(files=[('file',13,34)]))

    def test_aggregation_strict_threshold(self):
        rows=[dict(model_id='m',architecture='sd2',prompt_index=i,backdoor_probability=p,true_label=0)
              for i,p in enumerate([.3,.7])]
        self.assertEqual(aggregate_rows(rows,.5)[0]['stage1_prediction'],0)

    def test_failed_refresh_cannot_reuse_old_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folder=root/'model'; folder.mkdir()
            first=dict(model_id='model',architecture='sd2',stage1_score=.9,
                       stage1_threshold=.5,stage1_prediction=1,true_label=0,
                       train_validation_overlap='none')
            metric=dict(model_id='model',architecture='sd2',domain='gender',
                        hash_mean=.8,full_ssim_variance=.01,background_ssim_mean=.1)
            common.write_csv(root/'stage1.csv',[first])
            common.write_csv(folder/'model_metrics.csv',[metric])
            common.write_json(folder/'incomplete.json',dict(stage='measurement'))
            args=SimpleNamespace(output_dir=str(root),stage1_csv=str(root/'stage1.csv'),
                metrics=None,domain='gender',variance_weight=10.,threshold=4.,denominator_epsilon=1e-8)
            finalize(args)
            result=read_csv(root/'final_predictions.csv')[0]
            self.assertEqual(result['final_prediction'],'1')
            self.assertEqual(result['stage2_action'],'retain_unavailable')
            (folder/'incomplete.json').unlink()
            finalize(args)
            self.assertEqual(read_csv(root/'final_predictions.csv')[0]['final_prediction'],'0')


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch is not installed')
class Network(unittest.TestCase):
    def test_full_and_trajectory_free_forward_backward(self):
        import torch
        for traces in (list(TRACE_NAMES),[]):
            layout=dict(anchors=['a','b'],anchor_features=['x','y'],trajectory_features=traces)
            config=dict(layout=layout,hidden_dim=8,attention_heads=2,channel_layers=1,transformer_layers=1,dropout=0.)
            size=4+len(traces)+2
            net=detector.build_detector(size,3,config)
            data=torch.randn(2,3,size);data[...,-2:]=1
            output=net(data)
            self.assertEqual(tuple(output.shape),(2,))
            output.sum().backward()
            self.assertTrue(any(p.grad is not None for p in net.parameters()))


if __name__=='__main__':
    unittest.main()
