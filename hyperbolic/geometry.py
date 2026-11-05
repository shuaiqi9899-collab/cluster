"""Float64 Euclidean/Poincare geometry, curvature -1. No label inputs."""
import numpy as np
from scipy.spatial.distance import cdist


def ball_distance(x, y=None):
    x = np.asarray(x, dtype=np.float64)
    y = x if y is None else np.asarray(y, dtype=np.float64)
    nx = 1 - np.sum(x*x, axis=-1)
    ny = 1 - np.sum(y*y, axis=-1)
    if np.any(nx <= 0) or np.any(ny <= 0):
        raise ValueError('Point outside open unit ball')
    return 2*np.arcsinh(cdist(x, y)/np.sqrt(nx[:, None]*ny[None, :]))


def lorentz(x):
    s = np.sum(x*x, axis=-1, keepdims=True)
    return np.concatenate(((1+s)/(1-s), 2*x/(1-s)), axis=-1)


def mobius(x, y):
    xy = np.sum(x*y, axis=-1, keepdims=True)
    xx = np.sum(x*x, axis=-1, keepdims=True)
    yy = np.sum(y*y, axis=-1, keepdims=True)
    return ((1+2*xy+yy)*x+(1-xx)*y)/(1+2*xy+xx*yy)


def logmap(x, y):
    w = mobius(-x, y)
    n = np.linalg.norm(w, axis=-1, keepdims=True)
    ratio = np.divide(np.arctanh(np.minimum(n, 1-1e-15)), n,
                      out=np.ones_like(n), where=n > 0)
    return (1-np.sum(x*x, axis=-1, keepdims=True))*ratio*w


def expmap(x, v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ratio = np.divide(np.tanh(n/(1-np.sum(x*x, axis=-1, keepdims=True))), n,
                      out=np.ones_like(n), where=n > 0)
    return mobius(x, ratio*v)


def frechet(points, tolerance=1e-7, max_iterations=512):
    """Karcher mean with backtracking; fail explicitly on nonconvergence."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 1:
        return points[0].copy(), 0, 0.0
    m = points.mean(axis=0)
    objective = np.mean(ball_distance(m[None], points)**2)
    for it in range(max_iterations):
        tangent = logmap(m, points).mean(axis=0)
        residual = 2*np.linalg.norm(tangent)/(1-m@m)
        if residual <= tolerance:
            return m, it, float(residual)
        step = 1.0
        for _ in range(40):
            candidate = expmap(m, step*tangent)
            value = np.mean(ball_distance(candidate[None], points)**2)
            if value <= objective - 0.25*step*residual**2:
                break
            # Objective loses precision before the gradient does.
            if residual < 1e-6 and value <= objective + 1e-14:
                break
            step *= 0.5
        else:
            raise RuntimeError(f'Frechet line search failed; residual={residual}')
        m, objective = candidate, value
    raise RuntimeError(f'Frechet failed after {max_iterations}; residual={residual}')


class Geometry:
    def __init__(self, x, radius=0.0):
        self.x = np.asarray(x, dtype=np.float64)
        self.radius = float(radius)
        self.center = self.x.mean(axis=0)
        z = self.x-self.center
        self.R = float(np.linalg.norm(z, axis=1).max())
        if self.R == 0:
            raise ValueError('Degenerate dataset: all points coincide')
        if not 0 <= radius < 1:
            raise ValueError('radius must be in [0,1)')
        self.a = np.arctanh(radius)/self.R if radius else 1.0
        n = np.linalg.norm(z, axis=1, keepdims=True)
        factor = np.divide(np.tanh(self.a*n), n, out=np.full_like(n, self.a), where=n > 0)
        self.points = z*factor if radius else self.x.copy()
        de = cdist(self.x, self.x)
        self.distances = ball_distance(self.points) if radius else de
        np.fill_diagonal(self.distances, 0)
        tri = np.triu_indices(len(x), 1)
        qe = np.quantile(de[tri], .9)
        if qe <= 0:
            raise ValueError('Degenerate dataset: Q90 distance is zero')
        self.scale = float(np.quantile(self.distances[tri], .9)/qe)
        self._centers = {}
        self.mean_stats = dict(calls=0, iterations=0, max_residual=0.0)

    def mean(self, ids):
        ids = np.asarray(ids, dtype=int)
        key = ids.tobytes()
        if key not in self._centers:
            if self.radius:
                q, it, residual = frechet(self.points[ids])
                self.mean_stats['calls'] += 1
                self.mean_stats['iterations'] += it
                self.mean_stats['max_residual'] = max(self.mean_stats['max_residual'], residual)
            else:
                q = self.points[ids].mean(axis=0)
            self._centers[key] = q
        return self._centers[key]

    def chart(self, q):
        return self.center + logmap(np.zeros_like(q), q)/self.a if self.radius else q

    def distance(self, p, q):
        return float(ball_distance(p[None], q[None])[0, 0]) if self.radius else float(np.linalg.norm(p-q))
