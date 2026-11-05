"""Audited Fig.6 experiment. Original author files are imported, never modified."""
import argparse
import collections
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time
import traceback

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
# scikit-opt 0.6.6 sets the multiprocessing context at import. Import before
# creating workers, so its initialization is inherited instead of repeated.
import sko
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari

from .geometry import Geometry
from .prepare import ROOT, DATASETS, write_json

METHODS = ['MA-TD3', 'SA-TD3', 'Rand', 'BO-TPE', 'Anneal', 'PSO', 'GA', 'DE',
           'MA-FrozenActor', 'SA-FrozenActor', 'MA-Random']
SPACES = ['E', 'H', 'H-fixed-E', 'H-weak', 'H-strong']


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)


def load_geometry(out, dataset, seed, space):
    """Load canonical metric and reorder it, without recomputing global scales."""
    radius = {'E':0, 'H':.5, 'H-fixed-E':.5, 'H-weak':.2, 'H-strong':.8}[space]
    frozen = np.load(out/'inputs'/dataset/f'seed{seed}.npz')
    order = frozen['sample_ids']
    root = out/'geometry'/dataset/f'r{radius:g}'
    meta = json.loads((root/'mapping.json').read_text())
    g = Geometry.__new__(Geometry)
    g.x = frozen['x']; g.radius = radius
    g.center = np.asarray(meta['center']); g.R = meta['R']; g.a = meta['a']; g.scale = meta['scale']
    g.points = np.load(root/'points.npy')[order]
    g.distances = np.load(root/'distances.npy', mmap_mode='r')[np.ix_(order, order)]
    g._centers = {}; g.mean_stats = dict(calls=0, iterations=0, max_residual=0.)
    return g, frozen


def partition(g, seed):
    """Author entropy optimizer/k scan, exact deterministic metric kNN."""
    import torch
    sys.path.insert(0, str(ROOT/'author_source'))
    import utils.build_graph as bg
    import utils.pre_partition as pp
    seed_all(seed)
    metric = g.distances.copy()
    np.fill_diagonal(metric, np.inf)
    neighbors = np.argsort(metric, axis=1, kind='stable')
    audit = dict(scan=[])

    def graph(data, k):
        k = int(k)
        targets = np.repeat(np.arange(len(data)), k)
        sources = neighbors[:, :k].reshape(-1)
        edges = torch.from_numpy(np.column_stack((sources, targets)))
        lengths = torch.from_numpy(g.distances[sources, targets].copy())
        edges, lengths = bg.sumup_duplicates(edges, lengths)
        if lengths.mean() <= 0:
            raise ValueError('Degenerate zero-length kNN graph')
        weights = torch.exp2(-lengths/lengths.mean())
        audit['scan'].append(k)
        return edges, weights

    def aggregate(data, labels):
        values = labels.numpy().copy()
        sizes = np.bincount(values)
        small = np.flatnonzero(sizes <= 2)
        normal = np.flatnonzero(~np.isin(values, small))
        if len(normal) == 0:
            raise ValueError('No normal module available for small-module aggregation')
        for module in small:
            ids = np.flatnonzero(values == module)
            if len(ids):
                nearest = normal[np.argmin(g.distances[np.ix_(ids, normal)], axis=1)]
                values[ids] = values[nearest]
        _, values = np.unique(values, return_inverse=True)
        return torch.from_numpy(values)

    old_graph, old_pp_graph, old_aggregate = bg.build_knn_graph, pp.build_knn_graph, pp.aggregation_noise
    try:
        bg.build_knn_graph = pp.build_knn_graph = graph
        pp.aggregation_noise = aggregate
        modules, uncertainty, raw = pp.se_pre_partition(g.points, verbose=False)
    finally:
        bg.build_knn_graph, pp.build_knn_graph, pp.aggregation_noise = old_graph, old_pp_graph, old_aggregate
    uncertainty = uncertainty.numpy()
    groups = DBSCAN(eps=.3, min_samples=1).fit_predict(uncertainty.reshape(-1, 1))
    regions = groups[modules.numpy()]  # one-shot remapping; never mutate IDs in place
    return dict(modules=modules.numpy(), raw_tree=raw.numpy(), uncertainty=uncertainty,
                groups=groups, regions=regions, k=audit['scan'][-1], scanned_k=audit['scan'][:-1])


def get_partition(out, dataset, seed, space, g):
    key = 'E' if space == 'H-fixed-E' else space
    path = out/'partitions'/dataset/f'{key}_seed{seed}.npz'
    if path.exists():
        f = np.load(path)
        value = {k:f[k] for k in f.files}
        return value, float(value.get('preparation_seconds',0))
    started = time.perf_counter()
    if space == 'H-fixed-E':
        eg, _ = load_geometry(out, dataset, seed, 'E')
        value = partition(eg, seed)
    else:
        value = partition(g, seed)
    value['preparation_seconds'] = time.perf_counter()-started
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **value)
    return value, value['preparation_seconds']


def score_support(support_labels, mask, prediction):
    return float(nmi(support_labels, np.asarray(prediction)[mask])) if len(mask) >= 2 else 0.0


class LocalContext:
    """Contains positions and permitted labels only; no held-out labels."""
    def __init__(self, g, ids, mask, support_labels):
        self.g, self.ids, self.mask = g, np.asarray(ids), np.asarray(mask)
        self.support_labels = np.asarray(support_labels)
        self.features = g.points[ids]
        self.distances = g.distances[np.ix_(ids, ids)]
        self.mode = 'detect'
        self.events, self.fit_log, self.states = [], [], {}
        self.requests = collections.Counter()
        self.unique_fits = set()
        self.mode_seconds = collections.Counter()

    def fit(self, eps, m):
        start = time.perf_counter()
        eps, m = float(eps), int(m)
        labels = DBSCAN(eps=eps*self.g.scale, min_samples=m, metric='precomputed').fit_predict(self.distances)
        elapsed = time.perf_counter()-start
        key = (eps, m)
        self.unique_fits.add(key)
        self.requests[f'{self.mode}_fits'] += 1
        self.mode_seconds[self.mode] += elapsed
        self.fit_log.append(dict(mode=self.mode, u=eps, eps=eps*self.g.scale,
                                 min_samples=m, seconds=elapsed, fit_index=len(self.fit_log)+1))
        return labels

    def reward(self, labels):
        return score_support(self.support_labels, self.mask, labels)

    def state(self, labels, params, bump, bounds):
        key = np.asarray(labels, dtype=np.int64).tobytes()
        if key not in self.states:
            center = self.g.mean(self.ids)
            local = []
            for label in sorted(set(labels)):
                ids = self.ids[np.asarray(labels) == label]
                q = self.g.mean(ids)
                local.append([self.g.distance(q, center)/self.g.scale, len(ids)/100] + list(self.g.chart(q)))
            self.states[key] = local
        u, m = params
        glob = [u, u-bounds[0][0], bounds[0][1]-u, m/100,
                (m-bounds[1][0])/100, (bounds[1][1]-m)/100, len(set(labels))/len(labels)]
        if bump[0] == -1: glob[1] = 0
        elif bump[0] == 1: glob[2] = 0
        elif bump[1] == -1: glob[4] = 0
        elif bump[1] == 1: glob[5] = 0
        return [[glob], self.states[key]]

    def event(self, key, labels):
        self.events.append(dict(parameters=key, reward=self.reward(labels), labels=np.asarray(labels).copy(),
                                cumulative_fits=len(self.fit_log)))


def train_local(ctx, method, folder, region_id):
    """Released DrlDbscan train/detect; audited outer loop without full-label diagnostics."""
    import torch
    sys.path.insert(0, str(ROOT/'author_source'))
    import model.model as model
    from utils.utils import generate_parameter_space
    original = (model.DBSCAN, model.get_reward, model.get_state, model.ReplayBuffer)
    actor_steps, frozen_checks = [], []

    class MetricDBSCAN:
        def __init__(self, eps, min_samples): self.eps, self.m = eps, min_samples
        def fit_predict(self, features): return ctx.fit(self.eps, self.m)

    def reward(features, allowed, labels, count, n, mask, bump, buffer, step):
        value = ctx.reward(labels)
        return value, round(value, 4), value

    def state(features, labels, count, n, params, bump, bounds):
        return ctx.state(labels, params, bump, bounds)

    class Buffer(original[3]):
        def __init__(self, action_dim):
            # At most 14*29=406 transitions per layer in this fixed protocol.
            # Capacity reduction saves allocation, never wraps or changes sampling.
            super().__init__(action_dim, max_size=4096)
        def add(self, *args):
            if self.size >= self.max_size:
                raise RuntimeError('Replay capacity bound exceeded')
            return super().add(*args)

    class Cache(collections.OrderedDict):
        def __contains__(self, key):
            hit = super().__contains__(key)
            ctx.requests[f'{ctx.mode}_cache_hits' if hit else f'{ctx.mode}_cache_misses'] += 1
            return hit
        def __setitem__(self, key, value):
            fresh = not super().__contains__(key)
            super().__setitem__(key, value)
            if fresh and ctx.mode == 'train': ctx.event(key, value)

    try:
        model.DBSCAN, model.get_reward, model.get_state, model.ReplayBuffer = MetricDBSCAN, reward, state, Buffer
        sizes, steps, center, bounds = generate_parameter_space(ctx.features, 3, 5, 4, 'Shape')
        # Original integer step becomes zero only for extremely small regions.
        steps = [[s[0], max(1, s[1])] for s in steps]
        agents = [model.DrlDbscan(sizes, step, center, bounds, 'cpu', 16, 30, 2) for step in steps]
        actor_initial = []
        for agent in agents:
            initial = {k:v.detach().clone() for k,v in agent.agent.actor.state_dict().items()}
            actor_initial.append(initial)
            counter = [0]
            old_step = agent.agent.actor_optimizer.step
            if 'FrozenActor' in method:
                agent.agent.actor_optimizer.step = lambda *a, **kw: None
            else:
                def counted(*a, _counter=counter, _step=old_step, **kw):
                    _counter[0] += 1
                    return _step(*a, **kw)
                agent.agent.actor_optimizer.step = counted
            actor_steps.append(counter)
        detect_start = [0, center, 0]
        for agent in agents:
            agent.reset(detect_start)
            ctx.mode = 'detect'
            _, _, params = agent.detect(ctx.features, Cache())
            detect_start = [0, params[-1], 0]
        best = [0, center, 0]
        cache = Cache()
        allowed = np.full(len(ctx.ids), -987654321, dtype=int)
        allowed[ctx.mask] = ctx.support_labels
        episodes = []
        for layer, agent in enumerate(agents):
            agent.reset(best)
            history = [best[0]]
            old_size, improved_size = len(cache), len(cache)
            for episode in range(1, 15):
                agent.reset0()
                ctx.mode = 'train'
                result = agent.train(episode, ctx.mask, ctx.features, allowed, cache, .2,
                                     -1, [], log_flag=False)
                max_reward = result[4]
                if best[0] < max_reward[0]:
                    best = list(max_reward)
                    improved_size = len(cache)
                history.append(best[0])
                agent.reset0()
                ctx.mode = 'detect'
                agent.detect(ctx.features, cache)
                episodes.append(dict(layer=layer, episode=episode, best_support_reward=best[0],
                                     replay_size=agent.replay_buffer.size,
                                     critic_updates=agent.agent.total_iteration))
                if len(history) > 3 and history[-1] == history[-2] == history[-3] and history[-1] != history[0]:
                    break
            if improved_size == old_size: break
        for layer, agent in enumerate(agents):
            unchanged = all(torch.equal(v, actor_initial[layer][k]) for k,v in agent.agent.actor.state_dict().items())
            if 'FrozenActor' in method and not unchanged:
                raise AssertionError('Frozen actor weights changed')
            frozen_checks.append(unchanged)
        torch.save([dict(actor=a.agent.actor.state_dict(), critic=a.agent.critic.state_dict(),
                         critic_updates=a.agent.total_iteration, actor_updates=actor_steps[i][0],
                         actor_unchanged=frozen_checks[i], replay_size=a.replay_buffer.size)
                    for i,a in enumerate(agents)], folder/f'policies_region{region_id}.pt')
        return dict(episodes=episodes, critic_updates=[a.agent.total_iteration for a in agents],
                    actor_updates=[x[0] for x in actor_steps], actor_unchanged=frozen_checks)
    finally:
        model.DBSCAN, model.get_reward, model.get_state, model.ReplayBuffer = original


def baseline_local(ctx, method, seed):
    """Same six reconstructed optimizers; explicit MA random grid control."""
    from hyperopt import fmin, hp, rand, tpe, anneal
    n = len(ctx.ids)
    upper = max(1, math.ceil(n/4))
    eps_step, min_step = .008, max(1, math.ceil(int(n/4)/64))
    ctx.mode = 'train'
    def evaluate(params):
        eps = float(np.clip(round(float(params[0])/eps_step)*eps_step, eps_step, 1))
        m = int(np.clip(1+round((float(params[1])-1)/min_step)*min_step, 1, upper))
        labels = ctx.fit(eps, m)
        ctx.event(f'{eps}+{m}', labels)
        return -ctx.events[-1]['reward']
    if method == 'MA-Random':
        rng = np.random.default_rng(seed)
        initial_m = math.ceil(n/8)
        labels = ctx.fit(.5,initial_m)
        ctx.event(f'0.5+{initial_m}',labels)
        eps_grid = .5 + np.arange(-62,63)*eps_step
        m_grid = np.arange(initial_m%min_step or min_step,upper+1,min_step)
        if len(m_grid)==0: m_grid=np.array([initial_m])
        for _ in range(29):
            eps,m = float(rng.choice(eps_grid)),int(rng.choice(m_grid))
            labels=ctx.fit(eps,m)
            ctx.event(f'{eps}+{m}',labels)
    elif method in ['Rand', 'BO-TPE', 'Anneal']:
        algo = {'Rand':rand.suggest, 'BO-TPE':tpe.suggest, 'Anneal':anneal.suggest}[method]
        fmin(evaluate, [hp.uniform('eps', eps_step, 1), hp.uniform('MinPts', 1, upper)],
             algo=algo, max_evals=30, rstate=np.random.default_rng(seed), show_progressbar=False, verbose=False)
    else:
        class BudgetComplete(Exception): pass
        def objective(params):
            if len(ctx.events) >= 30: raise BudgetComplete()
            return evaluate(params)
        if method == 'PSO':
            from sko.PSO import PSO
            optimizer = PSO(objective, n_dim=2, pop=10, max_iter=10, lb=[eps_step,1], ub=[1,upper], w=.8,c1=.5,c2=.5)
        elif method == 'GA':
            from sko.GA import GA
            optimizer = GA(objective, n_dim=2, size_pop=10,max_iter=10,lb=[eps_step,1],ub=[1,upper],precision=[eps_step,min_step],prob_mut=.001)
        elif method == 'DE':
            from sko.DE import DE
            optimizer = DE(objective,n_dim=2,size_pop=10,max_iter=10,lb=[eps_step,1],ub=[1,upper],F=.5,prob_mut=.3)
        else: raise ValueError(method)
        try: optimizer.run()
        except BudgetComplete: pass
    if len(ctx.events) != 30: raise AssertionError('Baseline did not consume exactly 30 requests')
    return dict(actor_updates=[], critic_updates=[], actor_unchanged=[])


def merge(parts, frames, n):
    result = np.full(n, -1, dtype=int)
    offset = 0
    for ids, labels in zip(parts, frames):
        for value in sorted(set(labels)-{-1}):
            result[ids[np.asarray(labels) == value]] = offset
            offset += 1
    return result


def run_case(out, dataset, seed, space, method, g, frozen, pre, partition_seconds):
    folder = out/'cases'/space/f'{dataset}_{method}_seed{seed}'
    if (folder/'result.json').exists():
        old = json.loads((folder/'result.json').read_text())
        if old.get('status') == 'ok': return old
        raise RuntimeError(f'Failed case exists: {folder}; preserve it and choose a new output directory')
    folder.mkdir(parents=True, exist_ok=True)
    started, cpu_start = time.perf_counter(), time.process_time()
    seed_all(seed)
    g._centers = {}; g.mean_stats = dict(calls=0, iterations=0, max_residual=0.)
    is_ma = method.startswith('MA-')
    regions = pre['regions'] if is_ma else np.zeros(len(g.x), dtype=int)
    parts = [np.flatnonzero(regions == region) for region in sorted(set(regions))]
    reward_ids = frozen['reward_ids']
    y = frozen['y']
    contexts, trainings = [], []
    try:
        for region, ids in enumerate(parts):
            mask = np.flatnonzero(np.isin(ids, reward_ids))
            ctx = LocalContext(g, ids, mask, y[ids[mask]])
            with (folder/'run.log').open('a') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                info = train_local(ctx, method, folder, region) if method in ['MA-TD3','SA-TD3','MA-FrozenActor','SA-FrozenActor'] else baseline_local(ctx, method, seed)
            if not ctx.events: raise RuntimeError('No training candidates recorded')
            np.savez_compressed(folder/f'candidates_region{region}.npz', ids=ids, support_local_ids=mask,
                                labels=np.asarray([e['labels'] for e in ctx.events]),
                                rewards=np.asarray([e['reward'] for e in ctx.events]))
            write_json(folder/f'events_region{region}.json', [{k:v for k,v in e.items() if k != 'labels'} for e in ctx.events])
            write_json(folder/f'fits_region{region}.json', ctx.fit_log)
            info.update(size=len(ids), support_count=len(mask), support_class_count=len(np.unique(ctx.support_labels)),
                        unidentifiable=len(mask)<2, low_information=len(np.unique(ctx.support_labels))<2,
                        requests=dict(ctx.requests), unique_fits=len(ctx.unique_fits), events=len(ctx.events),
                        fit_seconds=dict(ctx.mode_seconds))
            trainings.append(info); contexts.append(ctx)
            # States are unnecessary after training; keep only lightweight candidate records.
            ctx.states.clear(); ctx.distances = None
            gc.collect()
        incumbents, raw, cumulative_cost = [], [], []
        best = [None]*len(contexts); rewards = [-np.inf]*len(contexts)
        for step in range(30):
            frames, costs = [], []
            for j, ctx in enumerate(contexts):
                event = ctx.events[min(step, len(ctx.events)-1)]
                frames.append(event['labels'])
                if event['reward'] > rewards[j]:
                    best[j], rewards[j] = event['labels'], event['reward']
                costs.append(event['cumulative_fits'])
            incumbents.append(merge(parts, best, len(y)))
            raw.append(merge(parts, frames, len(y)))
            cumulative_cost.append(sum(costs))
        predictions = np.asarray(incumbents)
        np.savez_compressed(folder/'predictions.npz', reward_incumbents=predictions, raw_rounds=raw,
                            cumulative_fits=cumulative_cost)
        held = np.ones(len(y), dtype=bool); held[reward_ids] = False
        full = [float(nmi(y, p)) for p in predictions]
        held_scores = [float(nmi(y[held], p[held])) for p in predictions]
        final = predictions[-1]
        diagnostics = {}
        for name, labels in [('raw_tree',pre['raw_tree']),('modules',pre['modules']),('regions',regions),('initial',raw[0])]:
            diagnostics[name] = dict(nmi=float(nmi(y,labels)), ari=float(ari(y,labels)))
        result = dict(status='ok', dataset=dataset, seed=seed, space=space, method=method,
                      radius=g.radius, scale=g.scale, curve=np.maximum.accumulate(full), round_nmi=full,
                      heldout_curve=held_scores, heldout_nmi30=held_scores[-1], heldout_ari30=float(ari(y[held],final[held])),
                      noise_rate=float(np.mean(final==-1)), clusters=len(set(final)-{-1}),
                      regions=trainings, diagnostics=diagnostics, partition_k=int(pre['k']),
                      largest_region_fraction=max(map(len, parts))/len(y),
                      partition_seconds_observed=partition_seconds, shared_partition=True,
                      cumulative_fits=cumulative_cost,
                      actual_fits=sum(len(c.fit_log) for c in contexts),
                      unique_fits=sum(len(c.unique_fits) for c in contexts),
                      actor_updates=sum(sum(i['actor_updates']) for i in trainings),
                      critic_updates=sum(sum(i['critic_updates']) for i in trainings),
                      frechet=g.mean_stats, seconds=time.perf_counter()-started,
                      cpu_seconds=time.process_time()-cpu_start,
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        result = dict(status='failed', dataset=dataset,seed=seed,space=space,method=method,
                      error=traceback.format_exc(),seconds=time.perf_counter()-started)
    write_json(folder/'result.json', result)
    return result


def methods_for(space):
    if space == 'H-fixed-E': return ['MA-TD3','MA-FrozenActor','MA-Random']
    if space in ['H-weak','H-strong']: return ['MA-TD3','SA-TD3','MA-FrozenActor','MA-Random']
    return METHODS


def group_worker(task):
    out, dataset, seed, space, chosen = task
    out = Path(out)
    g, frozen = load_geometry(out,dataset,seed,space)
    pre, seconds = get_partition(out,dataset,seed,space,g)
    results = []
    for method in chosen:
        result = run_case(out,dataset,seed,space,method,g,frozen,pre,seconds)
        results.append(result)
        print(f'{space} {dataset} seed{seed} {method}: {result["status"]} {result["seconds"]:.1f}s',flush=True)
        if result['status'] != 'ok': raise RuntimeError(result['error'])
    return [dict(space=r['space'],dataset=r['dataset'],seed=r['seed'],method=r['method'],status=r['status']) for r in results]


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/hyperbolic')
    parser.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(10)))
    parser.add_argument('--spaces',nargs='+',choices=SPACES,default=SPACES)
    parser.add_argument('--methods',nargs='+',choices=METHODS)
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    out = args.output.resolve()
    gate = json.loads((out/'validation/geometry_gate.json').read_text())
    if gate['status'] != 'passed': raise RuntimeError('Geometry gate has not passed')
    tasks = []
    for space in args.spaces:
        methods = [m for m in methods_for(space) if args.methods is None or m in args.methods]
        if not methods: continue
        for data in args.datasets:
            for seed in args.seeds: tasks.append((str(out),data,seed,space,methods))
    protocol = dict(version=1, git_base='5ed268983cf4b6b9f5660b895b29d79b2ce1621a',
                    cases=sum(len(t[-1]) for t in tasks), tasks=tasks, workers=args.workers,
                    support_fraction=.2, episodes=14,layers=3,step_limit=30,display_steps=30,
                    radii=[.2,.5,.8], frechet_tolerance=1e-7, primary_margin=.02,
                    input_origin='Regenerated with original loader semantics; historical input.npz unavailable',
                    ma_random='Exact initial center then 29 uniform draws on center-anchored final-step grid, no recursive window; historical audit code unavailable',
                    initialization='Fresh actor/critic per region/layer, same per-case seed',
                    labels='Training sees permitted support labels only; evaluation offline',
                    shared_setup='Geometry and partition cached across methods; report setup time separately',
                    replay_capacity=4096,replay_bound_per_layer=406)
    protocol['source_hashes'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (ROOT/'hyperbolic').glob('*.py')}
    run_id = time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    write_json(out/'invocations'/f'{run_id}.json',protocol)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(group_worker,t) for t in tasks]
        for f in as_completed(futures): f.result()
    print('All selected cases completed.',flush=True)


if __name__ == '__main__': main()
