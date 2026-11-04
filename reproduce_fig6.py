#!/usr/bin/env python
"""Observe released AR-DBSCAN without changing its search or training rules.

Run the author's complete default training, record newly cached training fits,
then align local partial-NMI incumbents and evaluate global historical maxima.
Baselines are separately labelled reconstructions, not released author code.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import collections
import contextlib
import csv
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import traceback
import types
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'author_source'
DATASETS = ['Aggregation', 'Compound', 'Pathbased', 'D31', 'Asymmetric',
            'Skewed', 'Cure-t1', 'Cure-t2']
METHODS = ['AR-DBSCAN', 'DRL-DBSCAN', 'Rand', 'BO-TPE', 'Anneal', 'PSO', 'GA', 'DE']


def dump(path, value):
    def serial(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(value, indent=2, default=serial, allow_nan=False))


def load_author():
    sys.path.insert(0, str(REPO))
    if 'model.model' in sys.modules:
        # Each process executes multiple cases; discard case-specific observers.
        importlib.reload(sys.modules['model.model'])
    spec = importlib.util.spec_from_file_location('released_ar_main', REPO / 'main.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Silence progress display only; retain its iterator/context-manager behavior.
    mod.tqdm = functools.partial(mod.tqdm, disable=True)
    mod.trange = functools.partial(mod.trange, disable=True)
    return mod


def merge(parts, frames, n):
    result = np.full(n, -1, dtype=np.int64)
    offset = 0
    for ids, frame in zip(parts, frames):
        labels = np.asarray(frame).copy()
        labels[labels != -1] += offset
        offset = int(labels.max()) + 1  # exactly main.py, including all-noise behavior
        result[ids] = labels
    return result


def author_run(mod, args, x, y, mask, method, folder):
    import torch
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import normalized_mutual_info_score as nmi
    if method == 'AR-DBSCAN':
        se, uncertainty, se_noise = mod.se_pre_partition(x, verbose=False)
        groups = DBSCAN(eps=0.3, min_samples=1).fit_predict(uncertainty.reshape(-1, 1))
        original_se = se.clone()
        # Retain released in-place remapping, including any ID collision.
        for i, group in enumerate(groups):
            se[se == i] = int(group)
        groups_n = int(groups.max()) + 1
        parts = [np.flatnonzero(se.numpy() == j) for j in range(groups_n)]
        local_masks = []
        original_ids = torch.arange(len(x))
        for j, ids in enumerate(parts):
            idx = (se == j).nonzero().squeeze()
            combined, counts = torch.cat((idx, torch.tensor(mask))).unique(return_counts=True)
            original_part = original_ids[se == j]
            local_mask = (combined[counts > 1].view(1, -1) == original_part.view(-1, 1)).any(dim=1).nonzero().squeeze()
            if local_mask.shape == torch.Size([]):
                local_mask = torch.tensor(random.sample(range(len(ids)), 2))
            elif local_mask.shape[0] == 1:
                local_mask = torch.cat((local_mask, local_mask))
            local_masks.append(local_mask)
        np.savez_compressed(folder / 'partition.npz', modules=original_se.numpy(),
                            regions=se.numpy(), uncertainty=uncertainty.numpy(),
                            groups=groups, raw_tree=se_noise.numpy())
        partition_info = dict(agents=groups_n, sizes=[len(p) for p in parts],
                              raw_tree_nmi=float(nmi(y, se_noise.numpy())),
                              remapping_collision=not np.array_equal(groups[original_se.numpy()], se.numpy()))
    else:
        parts, local_masks = [np.arange(len(x))], [mask]
        partition_info = dict(agents=1, sizes=[len(x)])

    traces, finals, policies = [], [], []
    active = dict(events=None, masks=None, labels=None, mode=None)

    class TraceCache(collections.OrderedDict):
        def __setitem__(self, key, value):
            fresh = key not in self
            super().__setitem__(key, value)
            if fresh and active['mode'] == 'train':
                score = float(nmi(active['labels'][active['masks']], np.asarray(value)[active['masks']]))
                active['events'].append(dict(parameters=key, reward=score,
                                             labels=np.asarray(value).copy()))

    original_train = mod.DrlDbscan.train
    original_detect = mod.DrlDbscan.detect
    original_init = mod.DrlDbscan.__init__

    def observe_train(self, episode_i, masks, features, labels, cache, *a, **kw):
        active.update(mode='train', masks=np.asarray(masks, dtype=int), labels=labels)
        try:
            return original_train(self, episode_i, masks, features, labels, cache, *a, **kw)
        finally:
            active['mode'] = None

    def observe_detect(self, *a, **kw):
        active['mode'] = 'detect'
        try: return original_detect(self, *a, **kw)
        finally: active['mode'] = None

    def observe_init(self, *a, **kw):
        original_init(self, *a, **kw)
        policies.append(self)

    mod.collections = types.SimpleNamespace(OrderedDict=TraceCache)
    mod.DrlDbscan.train, mod.DrlDbscan.detect = observe_train, observe_detect
    mod.DrlDbscan.__init__ = observe_init
    for j, (ids, local_mask) in enumerate(zip(parts, local_masks)):
        active['events'] = []
        labels, _ = mod.DRL_DBSCAN(args, [x[ids]], [y[ids]], [local_mask], j)
        finals.append(labels.copy())
        traces.append(active['events'])
    # Capture raw candidates as well as reward-selected trajectories for audit.
    payload = {}
    for j, events in enumerate(traces):
        if not events:
            raise RuntimeError(f'Agent {j} produced no training fits')
        payload[f'labels_{j}'] = np.asarray([e['labels'] for e in events])
        payload[f'rewards_{j}'] = np.asarray([e['reward'] for e in events])
    np.savez_compressed(folder / 'training_candidates.npz', **payload)
    dump(folder / 'training_events.json', [[{k:v for k,v in e.items() if k!='labels'} for e in events] for events in traces])
    torch.save([dict(actor=p.agent.actor.state_dict(), critic=p.agent.critic.state_dict(),
                     iterations=p.agent.total_iteration, replay_size=p.replay_buffer.size)
                for p in policies], folder / 'final_policies.pt')

    trajectories, best_frames, best_rewards = [], [None]*len(parts), [-np.inf]*len(parts)
    raw_trajectories = []
    for r in range(30):
        current = []
        for j, events in enumerate(traces):
            event = events[min(r, len(events)-1)]
            current.append(event['labels'])
            if event['reward'] > best_rewards[j]:
                best_rewards[j], best_frames[j] = event['reward'], event['labels']
        trajectories.append(merge(parts, best_frames, len(x)))
        raw_trajectories.append(merge(parts, current, len(x)))
    preds = np.asarray(trajectories)
    raw_preds = np.asarray(raw_trajectories)
    final = merge(parts, finals, len(x))
    np.savez_compressed(folder/'predictions.npz', reward_incumbents=preds,
                        raw_rounds=raw_preds, final_full_training=final)
    values = [float(nmi(y, p)) for p in preds]
    raw_values = [float(nmi(y, p)) for p in raw_preds]
    return dict(curve=np.maximum.accumulate(values).tolist(), round_nmi=values,
                raw_history_max=np.maximum.accumulate(raw_values).tolist(),
                final_full_training_nmi=float(nmi(y, final)),
                fits_per_agent=[len(t) for t in traces],
                critic_updates=sum(int(p.agent.total_iteration) for p in policies),
                replay_transitions=sum(int(p.replay_buffer.size) for p in policies),
                **partition_info)


def baseline_run(mod, args, x, y, mask, method, seed, folder):
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import normalized_mutual_info_score as nmi
    records, predictions = [], []
    rng = np.random.default_rng(seed)
    upper_m = int(np.ceil(len(x)/4))
    # Same released domain (Eps (0,1], MinPts [1,ceil(N/4)]) and last-layer steps.
    eps_step = 1 / 5**3
    m_step = int(np.ceil(int(len(x)/4)/4**3))

    def evaluate(params):
        eps = float(np.clip(round(float(params[0])/eps_step)*eps_step, eps_step, 1))
        m = int(np.clip(1 + round((float(params[1])-1)/m_step)*m_step, 1, upper_m))
        labels = DBSCAN(eps=eps, min_samples=m).fit_predict(x)
        reward = float(nmi(y[mask], labels[mask]))
        records.append(dict(eps=eps, min_samples=m, reward=reward))
        predictions.append(labels)
        return -reward

    if method in ('Rand', 'BO-TPE', 'Anneal'):
        from hyperopt import fmin, hp, rand, tpe, anneal
        algorithm = {'Rand':rand.suggest, 'BO-TPE':tpe.suggest, 'Anneal':anneal.suggest}[method]
        fmin(evaluate, [hp.uniform('eps', eps_step, 1), hp.uniform('MinPts', 1, upper_m)],
             algo=algorithm, max_evals=30, rstate=rng, show_progressbar=False, verbose=False)
    else:
        # Use the packages named by the paper. Unpublished hyperparameters are
        # explicit reconstruction choices; stop after exactly 30 objective calls.
        class BudgetComplete(Exception): pass
        def objective(params):
            if len(records) >= 30: raise BudgetComplete()
            return evaluate(params)
        if method == 'PSO':
            from sko.PSO import PSO
            optimizer = PSO(objective, n_dim=2, pop=10, max_iter=10,
                            lb=[eps_step,1], ub=[1,upper_m], w=.8, c1=.5, c2=.5)
        elif method == 'GA':
            from sko.GA import GA
            optimizer = GA(objective, n_dim=2, size_pop=10, max_iter=10,
                           lb=[eps_step,1], ub=[1,upper_m], precision=[eps_step,m_step], prob_mut=.001)
        elif method == 'DE':
            from sko.DE import DE
            optimizer = DE(objective, n_dim=2, size_pop=10, max_iter=10,
                           lb=[eps_step,1], ub=[1,upper_m], F=.5, prob_mut=.3)
        try: optimizer.run()
        except BudgetComplete: pass
    assert len(records) == 30
    best = np.maximum.accumulate([e['reward'] for e in records])
    selected, index, score = [], 0, -np.inf
    for i, e in enumerate(records):
        if e['reward'] > score: index, score = i, e['reward']
        selected.append(predictions[index])
    values = [float(nmi(y,p)) for p in selected]
    raw = [float(nmi(y,p)) for p in predictions]
    dump(folder/'training_events.json', records)
    np.savez_compressed(folder/'predictions.npz', raw_rounds=np.asarray(predictions), reward_incumbents=np.asarray(selected))
    return dict(curve=np.maximum.accumulate(values).tolist(), round_nmi=values,
                raw_history_max=np.maximum.accumulate(raw).tolist(), agents=1,
                fits_per_agent=[30], partial_reward_best=float(best[-1]))


def worker(task):
    dataset, method, seed, out = task
    folder = Path(out)/'cases'/f'{dataset}_{method}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        with (folder/'run.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            import torch
            torch.set_num_threads(1)
            mod = load_author()
            args = mod.parser.parse_args(['--dataset', 'Shape-'+dataset, '--seed',str(seed),'--device','cpu'])
            mod.set_seed(seed)
            mask, x, y = mod.load_data_shape(str(REPO/'data'/f'Shape-{dataset}.txt'), .2)
            mask = np.asarray(mask)
            np.savez_compressed(folder/'input.npz', x=x, y=y, reward_ids=mask)
            if method in ('AR-DBSCAN','DRL-DBSCAN'):
                result = author_run(mod,args,x,y,mask,method,folder)
            else:
                result = baseline_run(mod,args,x,y,mask,method,seed,folder)
        result.update(dataset=dataset,method=method,seed=seed,status='ok',seconds=time.perf_counter()-started)
    except Exception:
        result = dict(dataset=dataset,method=method,seed=seed,status='failed',
                      seconds=time.perf_counter()-started,error=traceback.format_exc())
    dump(folder/'result.json', result)
    return result


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = [json.loads(p.read_text()) for p in (out/'cases').glob('*/result.json')]
    valid = [r for r in results if r['status']=='ok']
    colors = ['#c7a43a','#666666','#3d71bb','#62b36b','#ed8b4b','#e76879','#8972c6','#99765c']
    summary = []
    for key, methods in [('author_only',METHODS[:2]),('with_reconstructed_baselines',METHODS)]:
        fig, axes = plt.subplots(2,4,figsize=(16,7.8),sharex=True,sharey=True)
        for ax, dataset in zip(axes.flat,DATASETS):
            for method,color in zip(METHODS,colors):
                if method not in methods: continue
                rows = [r for r in valid if r['dataset']==dataset and r['method']==method]
                if not rows: continue
                curves = np.array([r['curve'] for r in rows])
                mean, var = curves.mean(axis=0), curves.var(axis=0)
                label=method+(' *' if method not in METHODS[:2] else '')
                ax.plot(np.arange(1,31),mean,label=label,color=color,lw=1.8 if method in METHODS[:2] else 1.2)
                if method in ('AR-DBSCAN','DRL-DBSCAN','BO-TPE'):
                    ax.fill_between(np.arange(1,31),np.clip(mean-var,0,1),np.clip(mean+var,0,1),alpha=.17,color=color)
                if key=='with_reconstructed_baselines':
                    summary.append(dict(dataset=dataset,method=method,runs=len(rows),
                                        mean_nmi30=float(mean[-1]),variance_nmi30=float(var[-1]),
                                        mean_nmi1=float(mean[0]),mean_auc30=float(mean.mean())))
            ax.set_title(dataset); ax.grid(alpha=.25); ax.set_xlim(0,30); ax.set_ylim(0,1.02)
            ax.set_xlabel('Clustering round (new training fit / agent)')
        for ax in axes[:,0]: ax.set_ylabel('Historical maximum NMI')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=8,frameon=False,bbox_to_anchor=(.5,.965))
        fig.suptitle('Released AR-DBSCAN: Fig.6 reproduction attempt',y=.995,fontsize=15)
        fig.text(.5,.008,'Seeds 0–9; bands = mean ± variance; * reconstructed baselines; BDE unavailable. See protocol.json for counting ambiguities.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.035,1,.925])
        fig.savefig(out/f'fig6_{key}.png',dpi=180)
        fig.savefig(out/f'fig6_{key}.pdf')
        plt.close(fig)
    if summary:
        with (out/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    dump(out/'status.json',dict(total=len(results),ok=len(valid),failed=len(results)-len(valid),failures=[r for r in results if r['status']!='ok']))
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'formal')
    parser.add_argument('--datasets',nargs='+',default=DATASETS)
    parser.add_argument('--methods',nargs='+',default=METHODS)
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(10)))
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.plot_only:
        print(json.dumps(plot(args.output),indent=2));return
    dump(args.output/'protocol.json',dict(
        author_repo='https://github.com/RingBDStack/AR-DBSCAN',
        author_commit='b9cfcd41fc1d34348e75352ec457dc1d08df8e02',
        datasets=args.datasets,methods=args.methods,seeds=args.seeds,workers=args.workers,
        defaults=dict(train_size=.2,episode_num=15,layer_num=3,step_num=30,batch_size=16,eps_size=5,min_size=4,reward_factor=.2),
        round_definition='New entries in the released training label_dic only, including initial center. Cache hits and detect-only calls do not advance this axis. First 30 fits per agent, hold last if fewer.',
        selection='Local incumbent selected by 20% partial-label NMI; align agents; evaluate full-label NMI; cumulative historical maximum. Raw round curves also saved.',
        training='Complete released default training, including pre-training detection and inter-episode detection; first 30 training fits displayed retrospectively.',
        uncertainties=['Paper does not publish exact seeds, plotting code or raw Fig.6 curves.',
                       'Paper does not explicitly resolve cache hits, detection, initial center, and whether partial- or full-label maxima select displayed predictions.',
                       'Released in-place module ID remapping retained; collisions flagged.',
                       'Released all-noise offset handling retained, though it can reuse labels after an all-noise partition.',
                       'Six baselines reconstructed with hyperopt/scikit-opt; exact original settings unavailable; BDE not supplied and omitted.'],
        baseline_settings=dict(domain='Eps (0,1], MinPts [1,ceil(N/4)]',rounds=30,
                               eps_grid=.008,minpts_grid='ceil(floor(N/4)/64), anchored at 1',
                               hyperopt='0.2.7 defaults, including TPE default startup',population=10,
                               PSO=dict(w=.8,c1=.5,c2=.5),GA=dict(prob_mut=.001),DE=dict(F=.5,prob_mut=.3)),
        source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in REPO.rglob('*.py') if '.git' not in p.parts}))
    tasks=[(d,m,s,str(args.output)) for d in args.datasets for s in args.seeds for m in args.methods
           if not (args.output/'cases'/f'{d}_{m}_seed{s}'/'result.json').exists()]
    # This scikit-opt release sets multiprocessing's context at import time.
    # Import it once before creating the process pool; children inherit it.
    import sko
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result()
            print(f"{i}/{len(tasks)} {r['dataset']} {r['method']} seed={r['seed']} {r['status']} {r['seconds']:.1f}s",flush=True)
            if r['status']!='ok': print(r['error'],flush=True)
    print(json.dumps(plot(args.output),indent=2),flush=True)


if __name__=='__main__': main()
