"""Recompute saved support rewards and held-out incumbents independently."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi,adjusted_rand_score as ari
from .common import OUT,BACKENDS,methods,write,sha
from .experiment import merge


def verify_dataset(out,key,pilot=False):
 out=Path(out);root=out/('pilot/cases' if pilot else 'cases');raw=np.load(out/'inputs'/key/'raw.npz');y=raw['y'];valid=raw['label_valid'];count=rewards=0;errors=[]
 expected={(b,s,m) for b in BACKENDS for s in ([0] if pilot else range(10)) for m in methods(b)}
 found=set()
 for backend in BACKENDS:
  for file in sorted((root/backend/key).glob('seed*/*/result.json')):
   result=json.loads(file.read_text());identity=(backend,result['seed'],result['method']);found.add(identity)
   try:
    assert result['status']=='ok' and result['space']=='H' and result['backend']==backend
    if not pilot:assert result['training_signature']==json.loads((out/'protocol.json').read_text())['training_signature']
    folder=file.parent;prediction=np.load(folder/'predictions.npz');supports=np.load(out/'prepared'/key/f'seed{result["seed"]}.npz')['support_ids'];held=valid.copy();held[supports]=False
    np.testing.assert_array_equal(prediction['support_ids'],supports);np.testing.assert_array_equal(prediction['evaluation_ids'],np.flatnonzero(held))
    parts=[];frames=[];full=[];queries=fits=0
    selections=json.loads((folder/'selected_parameters.json').read_text())
    for i,info in enumerate(result['regions']):
     f=np.load(folder/f'candidates_region{i}.npz');ids=f['ids'];labels=f['labels'];scores=f['scores'];mask=f['support_local_ids']
     np.testing.assert_array_equal(mask,np.flatnonzero(np.isin(ids,supports)))
     events=json.loads((folder/f'events_region{i}.json').read_text());q=json.loads((folder/f'queries_region{i}.json').read_text());train=[a for a in q if a['mode']=='train']
     assert len(labels)==len(events)==len(train)==info['training_requests']
     assert info['physical_parameters']==len({json.dumps(e['parameters'],sort_keys=True) for e in q})
     assert sum(not e['cache_hit'] for e in q)==info['costs'].get('actual_fits',0)
     for j,e in enumerate(events):
      score=float(nmi(y[ids[mask]],labels[j,mask])) if len(mask)>=2 else 0.
      assert abs(score-e['score'])<1e-13 and abs(score-scores[j])<1e-13
      assert e['parameters']==train[j]['parameters'];assert 'n_clusters' not in e['parameters']
      if backend=='AC':assert e['parameters']['linkage'] in ['single','complete','average','ward']
      rewards+=1
     best=[int(np.argmax(scores[:min(step+1,len(scores))])) for step in range(30)]
     for step,index in enumerate(best):assert selections['display_steps'][step][i]['parameters']==events[index]['parameters']
     frames.append(labels[best]);parts.append(ids);full.append(labels[int(np.argmax(scores))])
     queries+=len(q);fits+=sum(not e['cache_hit'] for e in q)
     if result['method'] not in ['MA-TD3','SA-TD3','MA-FrozenActor','SA-FrozenActor']:assert len(events)==30
     if 'FrozenActor' in result['method']:assert not any(info['actor_updates']) and all(info['actor_unchanged'])
     if 'checkpoint' in info:
      p=folder/info['checkpoint'];assert sha(p)==info['checkpoint_sha256']
      with gzip.open(p,'rb') as archive:assert len(archive.read(4))==4
    assert sorted(np.concatenate(parts).tolist())==list(range(len(y)))
    assert queries==result['total_queries'] and fits==result['actual_fits']
    all_scores=[]
    for step in range(30):
     p=merge(parts,[v[step] for v in frames],len(y));np.testing.assert_array_equal(p,prediction['reward_incumbents'][step])
     assert abs(nmi(y[held],p[held])-result['heldout_curve'][step])<1e-13;all_scores.append(nmi(y[valid],p[valid]))
    np.testing.assert_allclose(np.maximum.accumulate(all_scores),result['curve'],atol=1e-13,rtol=0)
    p=merge(parts,full,len(y));np.testing.assert_array_equal(p,prediction['full_training_incumbent']);assert abs(nmi(y[held],p[held])-result['full_training_heldout_nmi'])<1e-13
    assert abs(ari(y[held],prediction['reward_incumbents'][-1][held])-result['heldout_ari30'])<1e-13
    count+=1
   except Exception as exc:
    import traceback
    errors.append(dict(case=str(file),error=traceback.format_exc()))
 geometry=json.loads((out/'prepared'/key/'geometry.json').read_text());assert geometry['space']=='H' and not geometry['euclidean_clustering'] and not geometry['euclidean_distance_matrix']
 points=np.load(out/'prepared'/key/'points.npy');distance=np.load(out/'prepared'/key/'distances.npy',mmap_mode='r');rng=np.random.default_rng(14)
 pairs=rng.integers(0,len(points),size=(1000,2));p,q=points[pairs[:,0]],points[pairs[:,1]]
 independent=np.arccosh(np.maximum(1.,1+2*np.sum((p-q)**2,axis=1)/((1-np.sum(p*p,axis=1))*(1-np.sum(q*q,axis=1)))))
 np.testing.assert_allclose(distance[pairs[:,0],pairs[:,1]],independent,atol=3e-8,rtol=1e-10)
 missing=sorted(expected-found)
 result=dict(status='passed' if not errors and not missing else 'failed',space='H',dataset=key,pilot=pilot,verified_cases=count,expected_cases=len(expected),independent_support_reward_checks=rewards,missing=missing,errors=errors)
 write(out/('pilot/validation' if pilot else 'validation')/(key+'.json'),result)
 if result['status']!='passed':raise RuntimeError(str(result)[:2000])
 return result


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--output',type=Path,default=OUT);p.add_argument('--pilot',action='store_true');a=p.parse_args();print(verify_dataset(a.output,a.dataset,a.pilot))
