"""Recompute rewards, incumbent selection and held-out metrics from artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi,adjusted_rand_score as ari
from hyperbolic.prepare import ROOT,write_json
from .experiment import merge
from .lifecycle import ACTIVE_BACKENDS,check_training_hashes


def verify(out):
    count=rewards=0;errors=[];source=ROOT/'results/hyperbolic'
    for path in sorted(out.glob('*/cases/*/*/result.json')):
        if path.parts[-5] not in ACTIVE_BACKENDS:continue
        r=json.loads(path.read_text())
        if r['status']!='ok':errors.append(str(path)+': failed');continue
        try:
            folder=path.parent;dataset=r['dataset'];seed=r['seed']
            frozen=np.load(source/'inputs'/dataset/f'seed{seed}.npz');support=frozen['sample_ids'][frozen['reward_ids']]
            y=np.load(source/'inputs'/dataset/'canonical.npz')['y'];held=np.ones(len(y),bool);held[support]=False
            prediction=np.load(folder/'predictions.npz');parts=[];frames=[];full=[];queries=0;fits=0
            choices=json.loads((folder/'selected_parameters.json').read_text())
            for region,info in enumerate(r['regions']):
                candidates=np.load(folder/f'candidates_region{region}.npz');ids=candidates['ids'];mask=candidates['support_local_ids']
                np.testing.assert_array_equal(mask,np.flatnonzero(np.isin(ids,support)))
                events=json.loads((folder/f'events_region{region}.json').read_text());q=json.loads((folder/f'queries_region{region}.json').read_text())
                labels=candidates['labels'];scores=candidates['scores'];train=[e for e in q if e['mode']=='train']
                assert len(events)==len(labels)==len(train)==info['training_requests']
                assert info['unique_physical_parameters']==len({json.dumps(e['parameters'],sort_keys=True) for e in q})
                assert sum(not e['cache_hit'] for e in q)==info['costs'].get('actual_fits',0)
                queries+=len(q);fits+=info['costs'].get('actual_fits',0)
                for i,e in enumerate(events):
                    score=float(nmi(y[ids[mask]],labels[i,mask])) if len(mask)>=2 else 0.
                    assert abs(score-e['score'])<1e-13 and abs(score-scores[i])<1e-13
                    assert e['parameters']==train[i]['parameters']
                    assert 'n_clusters' not in e['parameters']
                    rewards+=1
                indices=[int(np.argmax(scores[:min(s+1,len(scores))])) for s in range(30)]
                frames.append(labels[indices]);parts.append(ids);full.append(labels[int(np.argmax(scores))])
                for step,i in enumerate(indices):
                    assert choices['display_steps'][step][region]['parameters']==events[i]['parameters']
                if r['method'] not in ['MA-TD3','SA-TD3','MA-FrozenActor','SA-FrozenActor']:assert len(events)==30
                if 'FrozenActor' in r['method']:
                    assert all(info['actor_unchanged']) and not any(info['actor_updates'])
            assert queries==r['total_queries'] and fits==r['actual_fits']
            assert sorted(np.concatenate(parts).tolist())==list(range(len(y)))
            for step in range(30):
                merged=merge(parts,[f[step] for f in frames],len(y))
                np.testing.assert_array_equal(prediction['reward_incumbents'][step],merged)
                assert abs(nmi(y[held],merged[held])-r['heldout_curve'][step])<1e-13
            full=merge(parts,full,len(y));np.testing.assert_array_equal(full,prediction['full_training_incumbent'])
            assert abs(nmi(y[held],full[held])-r['full_training_heldout_nmi'])<1e-13
            assert abs(ari(y[held],merged[held])-r['heldout_ari30'])<1e-13
            assert len(set(merged)-{-1})==r['clusters']
            assert abs(np.mean(merged<0)-r['noise_fraction'])<1e-13
            count+=1
        except Exception as exc:
            import traceback
            errors.append(dict(path=str(path),error=traceback.format_exc()))
    old_sources=0
    for package in ['hyperbolic','ac_joint']:
        manifest=json.loads((ROOT/package/'report/source_manifest.json').read_text())
        for path,digest in manifest['source_sha256'].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:errors.append('Old source changed: '+path)
            old_sources+=1
    invocations=0
    for path in (out/'invocations').glob('*.json'):
        record=json.loads(path.read_text());hashes=record.get('source_hashes',{})
        for relative in check_training_hashes(record):
            errors.append(f'Training core differs from invocation {path.name}: {relative}')
        invocations+=1
    result=dict(status='passed' if not errors else 'failed',verified_cases=count,
                backend_scope=list(ACTIVE_BACKENDS),
                independent_support_reward_checks=rewards,unchanged_old_source_files=old_sources,
                training_core_invocations_verified=invocations,errors=errors)
    write_json(out/'validation.json',result);print(json.dumps(result),flush=True)
    if errors:raise RuntimeError('Artifact verification failed')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'results/density_cf');verify(p.parse_args().output.resolve())
