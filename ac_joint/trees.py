"""Native distance linkages and Riemannian Ward; no label inputs.

For H-Ward the CAT(0) variance inequality gives a lower bound on the
Fréchet-SSE merge cost. Lazy candidate evaluation avoids solving all unions.
Computed costs have explicit numerical error bounds; selection is to that
precision, not a claim of exact real arithmetic.
"""
import heapq
import time

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from hyperbolic.geometry import ball_distance, logmap, expmap

LINKAGES = ('single', 'complete', 'average', 'ward')


def mean_sse(points, initial=None, tolerance=1e-9, max_iterations=128):
    """Riemannian Newton solve using the exact constant-curvature Hessian."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 1:
        return points[0].copy(), 0., 0., 0
    m = points.mean(axis=0) if initial is None else np.asarray(initial).copy()
    eye = np.eye(points.shape[1])
    objective = float(np.mean(ball_distance(m[None], points)**2))
    for iteration in range(max_iterations):
        conformal = 2/(1-m@m)
        w = conformal*logmap(m, points)
        gradient = w.mean(axis=0)
        residual = float(np.linalg.norm(gradient))
        if residual <= tolerance:
            return m, objective*len(points), residual, iteration
        distance = np.linalg.norm(w, axis=1)
        unit = np.divide(w, distance[:, None], out=np.zeros_like(w), where=distance[:, None] > 0)
        beta = np.divide(distance, np.tanh(distance), out=np.ones_like(distance), where=distance > 0)
        hessian = beta.mean()*eye + np.einsum('n,ni,nj->ij', 1-beta, unit, unit)/len(points)
        direction = np.linalg.solve(hessian, gradient)
        step = 1.
        for _ in range(40):
            q = expmap(m, step*direction/conformal)
            value = float(np.mean(ball_distance(q[None], points)**2))
            new_residual = float(np.linalg.norm(logmap(q, points).mean(axis=0))*2/(1-q@q))
            sufficient = value <= objective-1e-4*step*float(gradient@direction)
            rounding = 64*np.finfo(float).eps*max(1., objective)
            if sufficient or (new_residual < residual/2 and value <= objective+rounding):
                break
            step *= .5
        else:
            raise RuntimeError(f'Fréchet Newton line search failed: {residual}')
        m, objective = q, value
    raise RuntimeError(f'Fréchet Newton did not converge: {residual}')


def riemannian_ward(points, distances, progress=None):
    n = len(points)
    if n < 2: return np.empty((0,4)), dict(merge_count=0, exact_union_evaluations=0)
    started = time.perf_counter()
    active = np.zeros(2*n-1, dtype=bool);active[:n] = True
    size = np.zeros(2*n-1,dtype=int);size[:n] = 1
    centers = np.zeros((2*n-1,points.shape[1]));centers[:n] = points
    sse = np.zeros(2*n-1);residuals = np.zeros(2*n-1)
    members = [np.array([i],dtype=int) for i in range(n)]+[None]*(n-1)
    condensed = squareform(distances,checks=False)
    costs = .5*condensed**2
    order = np.argsort(costs,kind='stable')
    starts = n*np.arange(n)-np.arange(n)*(np.arange(n)+1)//2
    pointer = 0
    heap, evaluated = [], {}
    rows=[];logs=[];solves=iterations=roundoff=0;max_residual=max_error=0.

    def initial_top():
        nonlocal pointer
        while pointer < len(order):
            index = int(order[pointer])
            a = int(np.searchsorted(starts,index,side='right')-1)
            b = int(a+1+index-starts[a])
            if active[a] and active[b]: return float(costs[index]),a,b
            pointer += 1
        return np.inf,-1,-1

    def clean():
        while heap and not(active[heap[0][1]] and active[heap[0][2]]):
            _,a,b,_=heapq.heappop(heap);evaluated.pop((a,b),None)

    for step in range(n-1):
        while True:
            clean()
            base = initial_top()
            next_entry = heap[0] if heap else (np.inf,-1,-1,False)
            if base <= next_entry[:3]:
                cost,a,b = base;pointer += 1
                q = expmap(centers[a],.5*logmap(centers[a],centers[b]))
                union_sse = float(np.sum(ball_distance(q[None],points[[a,b]])**2))
                residual = float(np.linalg.norm(logmap(q,points[[a,b]]).mean(axis=0))*2/(1-q@q))
                raw = union_sse
                error = 128*np.finfo(float).eps*max(1.,union_sse)
                break
            priority,a,b,exact = heapq.heappop(heap)
            if exact:
                q,union_sse,residual,raw,error = evaluated.pop((a,b))
                cost = max(raw,0.)
                break
            ids = np.sort(np.r_[members[a],members[b]])
            warm = expmap(centers[a],size[b]/(size[a]+size[b])*logmap(centers[a],centers[b]))
            q,union_sse,residual,it = mean_sse(points[ids],warm)
            solves += 1;iterations += it;max_residual=max(max_residual,residual)
            raw = float(union_sse-sse[a]-sse[b])
            # Strong convexity bounds objective error by n*residual**2;
            # the second term is a conservative float64 cancellation allowance.
            error = ((size[a]+size[b])*residual**2+size[a]*residuals[a]**2+size[b]*residuals[b]**2
                     +128*np.finfo(float).eps*(1+union_sse+sse[a]+sse[b]))
            if raw < -error: raise RuntimeError(f'Negative Ward increment beyond numerical bound: {raw}, {error}')
            if raw < 0: roundoff += 1
            max_error=max(max_error,error)
            evaluated[(a,b)] = (q,union_sse,residual,raw,error)
            heapq.heappush(heap,(max(raw,0.)+error,a,b,True))
        node = n+step
        active[a]=active[b]=False;active[node]=True
        size[node]=size[a]+size[b];centers[node]=q;sse[node]=union_sse;residuals[node]=residual
        members[node]=np.sort(np.r_[members[a],members[b]])
        members[a]=members[b]=None
        height=float(np.sqrt(2*max(raw,0.)))
        rows.append([a,b,height,int(size[node])])
        logs.append([float(raw),float(error),float(residual),float(union_sse)])
        others=np.flatnonzero(active[:node])
        if len(others):
            d=ball_distance(q[None],centers[others])[0]
            safe=np.maximum(d-residual-residuals[others],0.)
            lower=size[node]*size[others]/(size[node]+size[others])*safe**2
            lower=np.maximum(lower-128*np.finfo(float).eps*np.maximum(1.,lower),0.)
            for other,bound in zip(others,lower):
                heapq.heappush(heap,(float(bound),int(other),node,False))
        remaining=n-step-1
        if len(heap)>max(10000,3*remaining**2):
            heap=[entry for entry in heap if active[entry[1]] and active[entry[2]]]
            heapq.heapify(heap)
            evaluated={key:value for key,value in evaluated.items() if active[key[0]] and active[key[1]]}
        if progress and (step%250==0 or step==n-2):
            progress(dict(merges=step+1,n=n,union_solves=solves,seconds=time.perf_counter()-started))
    z=np.asarray(rows,dtype=float)
    return z,dict(merge_count=n-1,exact_union_evaluations=solves,newton_iterations=iterations,
                  max_mean_residual=max_residual,max_cost_error_bound=max_error,negative_roundoff_count=roundoff,
                  seconds=time.perf_counter()-started,merge_audit=np.asarray(logs),
                  selection='CAT(0) lower-bound lazy evaluation; computed costs ordered by conservative numerical upper estimates')


def effective_heights(z,n):
    values=np.zeros(2*n-1)
    inversions=0
    for i,row in enumerate(z):
        a,b=int(row[0]),int(row[1]);raw=row[2]
        if raw < max(values[a],values[b]): inversions += 1
        values[n+i]=max(raw,values[a],values[b])
    return values[n:],inversions


def make_tree(points,distances,method,hyperbolic=False,progress=None):
    started=time.perf_counter();n=len(points)
    if n<2: z=np.empty((0,4));meta={}
    elif method=='ward' and hyperbolic: z,meta=riemannian_ward(points,distances,progress)
    elif method=='ward': z=linkage(points,method='ward');meta={}
    else: z=linkage(squareform(distances,checks=False),method=method);meta={}
    heights,inversions=effective_heights(z,n)
    meta.update(method=method,n=n,hyperbolic=bool(hyperbolic),seconds=time.perf_counter()-started,
                inversions=inversions,height_convention='max(raw merge height, effective child heights)',
                root_height=float(heights[-1]) if len(heights) else 0.)
    return z,heights,meta


def cut_tree(z,heights,n,threshold):
    """Closed <= cut of a tree; handles ties and nonmonotone raw heights."""
    if n==1:return np.zeros(1,dtype=int)
    parent=np.arange(2*n-1)
    representatives=np.arange(2*n-1)
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return int(a)
    for i,row in enumerate(z):
        a,b=int(row[0]),int(row[1]);node=n+i
        representatives[node]=representatives[a]
        if heights[i]<=threshold:
            ra,rb=find(int(representatives[a])),find(int(representatives[b]))
            parent[rb]=ra
    labels=np.array([find(i) for i in range(n)])
    _,labels=np.unique(labels,return_inverse=True)
    return labels
