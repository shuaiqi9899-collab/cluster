"""One dataset/backend/seed/method case; full training, H-only evaluation."""
import argparse,gc,gzip,hashlib,json,os,random,shutil,time,traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import numpy as np
import torch
import sko  # Initialize scikit-opt's multiprocessing mode before workers.
from sklearn.metrics import normalized_mutual_info_score as nmi,adjusted_rand_score as ari
from ac_joint.search import train_local as categorical_train,baseline_local as categorical_baseline
from .search_db import train_local as db_train,baseline_local as db_baseline
from .common import OUT,BACKENDS,methods,write,sha
from .context import Bank,Context


def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)


def merge(parts,frames,n):
 result=np.full(n,-1,dtype=int);offset=0
 for ids,labels in zip(parts,frames):
  for label in sorted(set(labels)-{-1}):result[ids[labels==label]]=offset;offset+=1
 return result


def run_case(out,key,backend,seed,method,bank=None,pilot=False):
 out=Path(out)
 if backend not in BACKENDS:raise ValueError('Unauthorized backend; BIRCH remains retired')
 if method not in methods(backend):raise ValueError('Method not in frozen backend protocol')
 protocol=json.loads((out/'protocol.json').read_text()) if (out/'protocol.json').exists() else {}
 if not pilot and not protocol.get('training_signature'):raise RuntimeError('Start formal runs through benchmarks80.run to freeze the protocol first')
 folder=out/('pilot/cases' if pilot else 'cases')/backend/key/f'seed{seed}'/method
 if (folder/'result.json').exists():
  old=json.loads((folder/'result.json').read_text())
  if old['status']=='ok':
   if not pilot and old.get('training_signature')!=protocol['training_signature']:raise RuntimeError('Preserve case from different training signature')
   if not pilot and bank is not None and old.get('geometry_sha256')!=bank.geometry_sha256:raise RuntimeError('Preserve case from different geometry')
   return old
  raise RuntimeError('Failed case preserved: '+str(folder))
 folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter();cpu=time.process_time();seed_all(seed)
 bank=bank or Bank(out,key,backend);bank.mean_calls=0;bank.max_residual=0.
 frozen=np.load(bank.folder/f'seed{seed}.npz');supports=frozen['support_ids']
 data=np.load(out/'inputs'/key/'raw.npz');y=data['y'];valid=data['label_valid'];held=valid.copy();held[supports]=False
 keys=bank.index['regions'][str(seed)] if method.startswith('MA-') else [bank.index['global_key']]
 contexts=[];training=[]
 try:
  for region,digest in enumerate(keys):
   ids=bank.ids(digest);mask=np.flatnonzero(np.isin(ids,supports));ctx=Context(bank,digest,supports,y[ids[mask]])
   policy=folder/f'policies_region{region}.pt'
   if method in ['MA-TD3','SA-TD3','MA-FrozenActor','SA-FrozenActor']:
    info=db_train(ctx,method,seed,policy) if backend=='DBSCAN' else categorical_train(ctx,method,seed,policy)
    compressed=policy.with_suffix('.pt.gz')
    with policy.open('rb') as source,compressed.open('wb') as target:
     with gzip.GzipFile(fileobj=target,mode='wb',compresslevel=3,mtime=0) as archive:shutil.copyfileobj(source,archive)
    policy.unlink();info['checkpoint']=compressed.name;info['checkpoint_sha256']=sha(compressed)
   else:info=db_baseline(ctx,method,seed) if backend=='DBSCAN' else categorical_baseline(ctx,method,seed)
   np.savez_compressed(folder/f'candidates_region{region}.npz',ids=ids,support_local_ids=mask,
     labels=np.asarray([e['labels'] for e in ctx.events]),scores=[e['score'] for e in ctx.events])
   write(folder/f'events_region{region}.json',[{k:v for k,v in e.items() if k!='labels'} for e in ctx.events]);write(folder/f'queries_region{region}.json',ctx.queries)
   info.update(subset=digest,size=len(ids),support_count=len(mask),support_classes=len(np.unique(ctx.support_labels)),costs=dict(ctx.costs),
     encoded_parameters=len(ctx.encoded),physical_parameters=len(ctx.physical),distinct_partitions=len(ctx.clusterings),training_requests=len(ctx.events))
   contexts.append(ctx);training.append(info);ctx.states.clear();gc.collect()
  frames=[];raw=[];selections=[];best=[None]*len(contexts);scores=[-np.inf]*len(contexts)
  for step in range(30):
   current=[];choices=[]
   for i,ctx in enumerate(contexts):
    event=ctx.events[min(step,len(ctx.events)-1)];current.append(event['labels'])
    if event['score']>scores[i]:best[i]=event;scores[i]=event['score']
    choices.append({k:best[i][k] for k in ['coords','parameters','score']})
   parts=[c.ids for c in contexts];frames.append(merge(parts,[e['labels'] for e in best],len(y)));raw.append(merge(parts,current,len(y)));selections.append(choices)
  complete_events=[max(c.events,key=lambda e:e['score']) for c in contexts]
  complete=merge([c.ids for c in contexts],[e['labels'] for e in complete_events],len(y))
  frames=np.asarray(frames);all_nmi=[float(nmi(y[valid],p[valid])) for p in frames];held_nmi=[float(nmi(y[held],p[held])) for p in frames]
  final=frames[-1];nonnoise=final>=0;counts=np.bincount(final[nonnoise]) if np.any(nonnoise) else np.zeros(0,dtype=int)
  np.savez_compressed(folder/'predictions.npz',reward_incumbents=frames,raw_rounds=np.asarray(raw),full_training_incumbent=complete,support_ids=supports,evaluation_ids=np.flatnonzero(held))
  write(folder/'selected_parameters.json',dict(display_steps=selections,full_training=[{k:e[k] for k in ['coords','parameters','score']} for e in complete_events]))
  result=dict(status='ok',space='H',dataset=key,backend=backend,seed=seed,method=method,pilot=pilot,n=len(y),support_count=len(supports),evaluation_count=int(np.sum(held)),
    curve=np.maximum.accumulate(all_nmi),round_nmi=all_nmi,heldout_curve=held_nmi,heldout_nmi30=held_nmi[-1],heldout_ari30=float(ari(y[held],final[held])),
    full_training_heldout_nmi=float(nmi(y[held],complete[held])),noise_fraction=float(np.mean(~nonnoise)),clusters=len(counts),
    singleton_fraction=float(np.sum(counts==1)/len(y)),largest_cluster_fraction=float(counts.max()/len(y)) if len(counts) else 0.,
    regions=training,agents=len(contexts),all_training_requests=sum(len(c.events) for c in contexts),total_queries=sum(len(c.queries) for c in contexts),
    actual_fits=sum(c.costs['actual_fits'] for c in contexts),physical_parameters=sum(len(c.physical) for c in contexts),distinct_partitions=sum(len(c.clusterings) for c in contexts),
    actor_updates=sum(sum(t['actor_updates']) for t in training),critic_updates=sum(sum(t['critic_updates']) for t in training),
    state_mean_calls=bank.mean_calls,max_mean_residual=bank.max_residual,seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
    raw_sha256=bank.index['data_sha256'],representation=bank.index['representation'],preparation_shared=True)
 except Exception:
  result=dict(status='failed',space='H',dataset=key,backend=backend,seed=seed,method=method,pilot=pilot,error=traceback.format_exc(),seconds=time.perf_counter()-start)
 result['training_signature']=protocol.get('training_signature','unfrozen-pilot');result['geometry_sha256']=bank.geometry_sha256
 write(folder/'result.json',result)
 if result['status']!='ok':raise RuntimeError(result['error'])
 return result


def run_group(out,key,backend,seed,chosen=None,pilot=False):
 bank=Bank(out,key,backend);results=[]
 for method in chosen or methods(backend):
  r=run_case(out,key,backend,seed,method,bank,pilot);print(key,backend,seed,method,r['status'],f'{r["seconds"]:.2f}s',flush=True);results.append(r)
 return results


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--backend',choices=BACKENDS,required=True);p.add_argument('--seed',type=int,required=True)
 p.add_argument('--methods',nargs='+');p.add_argument('--output',type=Path,default=OUT);p.add_argument('--pilot',action='store_true');a=p.parse_args()
 if a.seed not in range(10):p.error('Only seeds0..9 are permitted')
 run_group(a.output,a.dataset,a.backend,a.seed,a.methods,a.pilot)
