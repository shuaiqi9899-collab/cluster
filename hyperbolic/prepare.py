"""Freeze canonical sample IDs and run geometric/readiness gates before training."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler

from .geometry import Geometry, ball_distance, lorentz, logmap, expmap, frechet
from .boundary import control_edge

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['Aggregation', 'Compound', 'Pathbased', 'D31', 'Asymmetric', 'Skewed', 'Cure-t1', 'Cure-t2']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    def cast(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=cast, allow_nan=False))


def prepare(out):
    reports = []
    for name in DATASETS:
        raw = np.loadtxt(ROOT/'author_source/data'/f'Shape-{name}.txt')
        x = MinMaxScaler().fit_transform(raw[:, :2])
        y = raw[:, 2].astype(int)
        n = len(x)
        path = out/'inputs'/name
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path/'canonical.npz', x=x, y=y, sample_ids=np.arange(n))
        for seed in range(10):
            rng = random.Random(seed)
            order = np.asarray(rng.sample(range(n), n))
            ids = np.asarray(rng.sample(range(n), int(.2*n)))
            # Same Python random.sample calls as the original loader.
            np.savez_compressed(path/f'seed{seed}.npz', x=x[order], y=y[order],
                                sample_ids=order, reward_ids=ids)
        base = Geometry(x)
        de = base.distances
        tri = np.triu_indices(n, 1)
        # Exact isometry control, using actual ball coordinates.
        z = x-x.mean(axis=0)
        a = 1/np.linalg.norm(z, axis=1).max()
        v = a*z
        vv = np.sum(v*v, axis=1, keepdims=True)
        p3 = np.concatenate((2*v, vv), axis=1)/(vv+4)
        dh3 = ball_distance(p3)
        exact = 2*np.arcsinh(a*de/2)
        error = float(np.max(np.abs(dh3-exact)))
        assert error < 1e-12, (name, error)
        replay = []
        # Meaningful fixed probes; subsequently replay actual training candidates.
        for eps, m in [(.008, 2), (.04, 5), (.2, max(2, int(np.ceil(n/8)))), (.5, max(2, int(np.ceil(n/4))))]:
            eh = float(2*np.arcsinh(a*eps/2))
            be = de <= eps
            bh = dh3 <= eh
            ii, jj = np.where(np.triu(be != bh, 1))
            boundary = [dict(i=int(i), j=int(j), euclidean_gap=float(de[i,j]-eps),
                             hyperbolic_gap=float(dh3[i,j]-eh)) for i,j in zip(ii,jj)]
            # Preserve every raw discrepancy; never enlarge eps or hide it using allclose.
            if boundary:
                write_json(out/'validation'/f'{name}_boundary_eps{eps}.json', boundary)
                print(name, 'raw threshold boundary discrepancies:', len(boundary), flush=True)
                if any(abs(b['euclidean_gap']) > 1e-12 or abs(b['hyperbolic_gap']) > 1e-12 for b in boundary):
                    raise AssertionError('Non-boundary adjacency discrepancy')
            e = DBSCAN(eps=eps, min_samples=m, metric='precomputed').fit(de)
            h = DBSCAN(eps=eh, min_samples=m, metric='precomputed').fit(dh3)
            assert np.array_equal(e.core_sample_indices_, h.core_sample_indices_)
            assert np.array_equal(e.labels_ == -1, h.labels_ == -1)
            assert adjusted_rand_score(e.labels_, h.labels_) == 1
            near = (np.abs(de-eps) <= 64*np.finfo(float).eps*max(1,eps)) | (be != bh)
            ni,nj = np.where(np.triu(near,1))
            corrected = []
            for i,j in zip(ni,nj):
                exact_e, exact_h = control_edge(x[i],x[j],x.mean(axis=0),a,eps)
                corrected.append(dict(i=int(i),j=int(j), exact_euclidean=exact_e,exact_hyperbolic=exact_h,
                                      float_euclidean=bool(be[i,j]),float_hyperbolic=bool(bh[i,j])))
                be[i,j] = be[j,i] = exact_e
                bh[i,j] = bh[j,i] = exact_h
            assert np.array_equal(be,bh)
            ce = DBSCAN(eps=1,min_samples=m,metric='precomputed').fit(np.where(be,0.,2.))
            ch = DBSCAN(eps=1,min_samples=m,metric='precomputed').fit(np.where(bh,0.,2.))
            assert np.array_equal(ce.core_sample_indices_,ch.core_sample_indices_)
            assert np.array_equal(ce.labels_==-1,ch.labels_==-1)
            assert adjusted_rand_score(ce.labels_,ch.labels_) == 1
            replay.append(dict(eps=eps, min_samples=m, adjacency_equal=not boundary,
                               raw_boundary_discrepancies=boundary, ari=1.0,
                               exact_boundary_rechecks=corrected, exact_adjacency_equal=True,
                               core_points=len(e.core_sample_indices_), clusters=len(set(e.labels_)-{-1})))
        for radius in [0, .2, .5, .8]:
            g = base if radius == 0 else Geometry(x, radius)
            dest = out/'geometry'/name/f'r{radius:g}'
            dest.mkdir(parents=True, exist_ok=True)
            np.save(dest/'distances.npy', g.distances)
            np.save(dest/'points.npy', g.points)
            write_json(dest/'mapping.json', dict(radius=radius, a=g.a, R=g.R,
                                                center=g.center, scale=g.scale))
            if radius == 0: continue
            q = lorentz(g.points)
            # Independent Lorentz formula on a deterministic subset (avoid NxNx3 allocation).
            sample = np.linspace(0, n-1, min(n, 256), dtype=int)
            t = q[sample]
            ld = np.arccosh(np.maximum(t[:, :1]*t[:, 0][None]-t[:, 1:]@t[:, 1:].T, 1))
            bd = g.distances[np.ix_(sample, sample)]
            off = ~np.eye(len(sample), dtype=bool)
            lorentz_error = float(np.max(np.abs(ld[off]-bd[off])))
            assert lorentz_error < 1e-7
            assert np.isfinite(g.distances).all()
            assert np.max(np.abs(g.distances-g.distances.T)) < 1e-12
            assert np.count_nonzero(np.diag(g.distances)) == 0
            assert np.linalg.norm(g.points, axis=1).max() <= radius+1e-14
            roundtrip = float(np.max(np.abs(g.chart(g.points)-x)))
            assert roundtrip < 1e-12
            assert np.max(np.abs(expmap(np.zeros(2), logmap(np.zeros(2), g.points))-g.points)) < 1e-12
            mean, iterations, residual = frechet(g.points)
            assert np.linalg.norm(logmap(mean, g.points).mean(axis=0))*2/(1-mean@mean) < 1e-7
            neighbor = {}
            for k in [10, 30]:
                ee, hh = de.copy(), g.distances.copy()
                np.fill_diagonal(ee, np.inf); np.fill_diagonal(hh, np.inf)
                ei = np.argsort(ee, axis=1, kind='stable')[:, :k]
                hi = np.argsort(hh, axis=1, kind='stable')[:, :k]
                overlap = np.mean([len(set(e)&set(h))/k for e, h in zip(ei, hi)])
                local_ratio = g.distances[np.arange(n), hi[:, -1]]/(g.scale*de[np.arange(n), ei[:, -1]])
                neighbor[str(k)] = dict(overlap=float(overlap), normalized_radius_ratio_quantiles=np.quantile(local_ratio,[0,.1,.5,.9,1]))
            ratios = g.distances[tri]/(g.scale*np.maximum(de[tri], 1e-300))
            ratios = ratios[de[tri] > 0]
            changes = []
            for row in replay:
                eps, m = row['eps'], row['min_samples']
                be, bh = de <= eps, g.distances <= g.scale*eps
                changes.append(dict(eps=eps, min_samples=m,
                                    changed_pair_fraction=float(np.mean(be[tri] != bh[tri])),
                                    changed_core_fraction=float(np.mean((be.sum(axis=1)>=m)!=(bh.sum(axis=1)>=m)))))
            report = dict(dataset=name, radius=radius, n=n, scale=g.scale,
                          rank_correlation=float(spearmanr(de[tri], g.distances[tri]).statistic),
                          normalized_distance_ratio_quantiles=np.quantile(ratios,[0,.1,.5,.9,1]),
                          roundtrip_error=roundtrip, lorentz_error=lorentz_error,
                          frechet_iterations=iterations, frechet_residual=residual,
                          neighbors=neighbor, threshold_changes=changes)
            reports.append(report)
            print(name, radius, 'geometry passed', flush=True)
        write_json(out/'validation'/f'{name}_isometry.json', dict(max_distance_error=error, probes=replay))
    write_json(out/'validation/geometry.json', reports)
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(ROOT.rglob('*.py')) if '.venv' not in p.parts and 'results' not in p.parts}
    write_json(out/'source_hashes.json', hashes)
    write_json(out/'validation/geometry_gate.json', dict(status='passed', datasets=DATASETS, mappings=[.2,.5,.8]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'results/hyperbolic')
    args = parser.parse_args()
    prepare(args.output.resolve())
