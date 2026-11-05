"""HDBSCAN/BIRCH experiments; prior DBSCAN and AC artifacts are read-only inputs."""
import argparse
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import time
import traceback

for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[variable]='1'
import sko  # initialize scikit-opt's fork context before workers
import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score as nmi,adjusted_rand_score as ari

from hyperbolic.prepare import ROOT,DATASETS,write_json
from .context import Bank,Context
from ac_joint.search import train_local,baseline_local


METHODS=['MA-TD3','SA-TD3','Rand','BO-TPE','Anneal','PSO','GA','DE',
         'MA-FrozenActor','SA-FrozenActor','MA-Random','Grid-30']
SPACES=['E','H','H-fixed-E','H-weak','H-strong']


def methods_for(space):
    if space=='H-fixed-E':return ['MA-TD3','MA-FrozenActor','MA-Random']
    if space in ['H-weak','H-strong']:return ['MA-TD3','SA-TD3','MA-FrozenActor','MA-Random']
    return METHODS


def merge(parts,frames,n):
    result=np.full(n,-1,dtype=int);offset=0
    for ids,labels in zip(parts,frames):
        labels=np.asarray(labels);valid=labels>=0
        result[np.asarray(ids)[valid]]=labels[valid]+offset
        if np.any(valid):offset+=int(labels[valid].max())+1
    return result


def run_case(source,out,bank,dataset,space,seed,method):
    folder=out/bank.backend/'cases'/space/f'{dataset}_{method}_seed{seed}'
    if (folder/'result.json').exists():
        result=json.loads((folder/'result.json').read_text())
        if result['status']=='ok':return result
        raise RuntimeError(f'Preserve failed case and select a fresh output: {folder}')
    folder.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();cpu=time.process_time()
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    frozen=np.load(source/'inputs'/dataset/f'seed{seed}.npz')
    support_ids=frozen['sample_ids'][frozen['reward_ids']]
    y=np.load(source/'inputs'/dataset/'canonical.npz')['y']
    region_space='E' if space=='H-fixed-E' else space
    keys=bank.index['regions'][f'{region_space}_seed{seed}'] if method.startswith('MA-') else [bank.index['global_key']]
    bank.center_cache={};bank.mean_calls=0;bank.max_residual=0.
    contexts=[];training=[]
    try:
        for region,key in enumerate(keys):
            ids=bank.ids(key);mask=np.flatnonzero(np.isin(ids,support_ids))
            ctx=Context(bank,key,support_ids,y[ids[mask]])
            if method in ['MA-TD3','SA-TD3','MA-FrozenActor','SA-FrozenActor']:
                info=train_local(ctx,method,seed,folder/f'policies_region{region}.pt')
            else:info=baseline_local(ctx,method,seed)
            np.savez_compressed(folder/f'candidates_region{region}.npz',ids=ids,support_local_ids=mask,
                                labels=np.asarray([e['labels'] for e in ctx.events]),scores=[e['score'] for e in ctx.events])
            write_json(folder/f'events_region{region}.json',[{k:v for k,v in e.items() if k!='labels'} for e in ctx.events])
            write_json(folder/f'queries_region{region}.json',ctx.queries)
            info.update(ids_key=key,size=len(ids),support_count=len(mask),
                        support_class_count=len(np.unique(ctx.support_labels)),unidentifiable=len(mask)<2,
                        costs=dict(ctx.costs),unique_physical_parameters=len(ctx.physical_pairs),unique_parameter_pairs=len(ctx.distinct_pairs),
                        unique_clusterings=len(ctx.distinct_predictions),training_requests=len(ctx.events))
            contexts.append(ctx);training.append(info)
            ctx.state_cache.clear();ctx.partitions.clear();gc.collect()
        parts=[ctx.ids for ctx in contexts];best=[None]*len(parts);best_score=[-np.inf]*len(parts)
        frames=[];raw=[];cuts=[];queries=[];selected=[]
        for step in range(30):
            current=[];stepcuts=stepqueries=0;choices=[]
            for i,ctx in enumerate(contexts):
                event=ctx.events[min(step,len(ctx.events)-1)];current.append(event['labels'])
                if event['score']>best_score[i]:best[i]=event;best_score[i]=event['score']
                choices.append({k:best[i][k] for k in ['category_id','category_value','u','parameters','score']})
                stepcuts+=event['actual_fits'];stepqueries+=event['cumulative_queries']
            frames.append(merge(parts,[e['labels'] for e in best],len(y)))
            raw.append(merge(parts,current,len(y)));cuts.append(stepcuts);queries.append(stepqueries);selected.append(choices)
        full=[];fullchoices=[]
        for ctx in contexts:
            event=max(ctx.events,key=lambda e:e['score']);full.append(event['labels'])
            fullchoices.append({k:event[k] for k in ['category_id','category_value','u','parameters','score']})
        complete=merge(parts,full,len(y))
        prediction=np.asarray(frames);held=np.ones(len(y),dtype=bool);held[support_ids]=False
        full_scores=[float(nmi(y,p)) for p in prediction]
        held_scores=[float(nmi(y[held],p[held])) for p in prediction]
        np.savez_compressed(folder/'predictions.npz',reward_incumbents=prediction,raw_rounds=raw,
                            full_training_incumbent=complete,actual_cuts_at_display=cuts,total_queries_at_display=queries)
        write_json(folder/'selected_parameters.json',dict(display_steps=selected,full_training=fullchoices))
        final=prediction[-1]
        result=dict(status='ok',dataset=dataset,space=space,seed=seed,method=method,
                    curve=np.maximum.accumulate(full_scores),round_nmi=full_scores,heldout_curve=held_scores,
                    heldout_nmi30=held_scores[-1],heldout_ari30=float(ari(y[held],final[held])),
                    full_training_heldout_nmi=float(nmi(y[held],complete[held])),
                    backend=bank.backend,noise_fraction=float(np.mean(final<0)),clusters=int(len(set(final)-{-1})),singleton_fraction=float(np.mean([np.sum(final==label)==1 if label>=0 else False for label in final])),
                    largest_cluster_fraction=float(max([np.sum(final==label) for label in set(final)-{-1}]+[0])/len(y)),agents=len(parts),
                    regions=training,selected_parameters=selected[-1],
                    all_training_requests=sum(len(c.events) for c in contexts),
                    total_queries=sum(len(c.queries) for c in contexts),actual_fits=sum(c.costs['actual_fits'] for c in contexts),
                    unique_parameter_pairs=sum(len(c.distinct_pairs) for c in contexts),
                    unique_clusterings=sum(len(c.distinct_predictions) for c in contexts),
                    actor_updates=sum(sum(t['actor_updates']) for t in training),
                    critic_updates=sum(sum(t['critic_updates']) for t in training),
                    mean_calls=bank.mean_calls,max_mean_residual=bank.max_residual,
                    actual_cuts_at_display=cuts,total_queries_at_display=queries,
                    seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                    process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    cache_shared=True,hierarchy_and_fit_costs_in_regions=True)
    except Exception:
        result=dict(status='failed',dataset=dataset,space=space,seed=seed,method=method,
                    seconds=time.perf_counter()-started,error=traceback.format_exc())
    write_json(folder/'result.json',result)
    return result


def group_worker(task):
    source,ac_source,out,backend,dataset,space,seed,methods=task;source,out=Path(source),Path(out)
    bank=Bank(source,ac_source,out,dataset,space,backend,seed)
    results=[]
    for method in methods:
        r=run_case(source,out,bank,dataset,space,seed,method)
        print(f'{backend} {space} {dataset} seed{seed} {method}: {r["status"]} {r["seconds"]:.2f}s',flush=True)
        if r['status']!='ok':raise RuntimeError(r['error'])
        results.append(dict(space=space,dataset=dataset,seed=seed,method=method,status='ok'))
    return results


def main():
    from concurrent.futures import ProcessPoolExecutor,as_completed
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'results/hyperbolic')
    parser.add_argument('--output',type=Path,default=ROOT/'results/density_cf')
    parser.add_argument('--ac-source',type=Path,default=ROOT/'results/ac_joint')
    parser.add_argument('--backends',nargs='+',choices=['HDBSCAN','BIRCH'],default=['HDBSCAN','BIRCH'])
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    parser.add_argument('--spaces',nargs='+',choices=SPACES,default=SPACES)
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(10)))
    parser.add_argument('--methods',nargs='+',choices=METHODS)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    tasks=[]
    for backend in args.backends:
        for space in args.spaces:
            chosen=[m for m in methods_for(space) if args.methods is None or m in args.methods]
            if not chosen:continue
            for d in args.datasets:
                for seed in args.seeds:tasks.append((str(source),str(args.ac_source.resolve()),str(out),backend,d,space,seed,chosen))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'density_cf').glob('*.py')}
    protocol=dict(version=1,cases=sum(len(t[-1]) for t in tasks),tasks=tasks,workers=args.workers,
                  search_parameters={'HDBSCAN':['min_samples in [1,5,10,20], capped n-1','log min_cluster_size: 2..ceil(n_local/4)'],
                                     'BIRCH':['branching_factor in [10,25,50,100]','log RMS radius: global delta/(10 sqrt(N))..Dmax']},
                  global_state=8,local_state=4,actions=7,layers=3,max_episodes=14,max_episode_steps=30,
                  batch_size=16,display_training_requests=30,
                  training_budget='Complete training; retrospective first30 training requests incl4 warmups; all detect/training/cache costs retained',
                  hdbscan='Pinned0.8.40, precomputed native metric, EOM, alpha1, epsilon0, allow_single_clusterFalse; min_samples excludes self',
                  birch='E sklearn Birch n_clusters=None; H intrinsic CF Fréchet SSE/N, retained members, native split/assignment; identical frozen stream order',
                  initial_probes='Four categories at u=.5, order rotated by seed; included in first30',
                  baselines='Categorical enumeration for PSO/GA/DE, pop4, quotas8/8/7/7 rotatedseed; categorical Hyperopt',
                  source_hashes=hashes)
    write_json(out/'invocations'/f'{time.strftime("%Y%m%d_%H%M%S",time.gmtime())}.json',protocol)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(group_worker,t) for t in tasks]):future.result()
    print('All selected density/CF cases completed.',flush=True)


if __name__=='__main__':main()
