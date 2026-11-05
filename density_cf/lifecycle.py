"""Active scope and integrity checks for the preserved v1 training engine."""
import hashlib
from hyperbolic.prepare import ROOT

ACTIVE_BACKENDS=('HDBSCAN',)
RETIRED_BACKENDS=('BIRCH',)
FROZEN_ENGINE='density_cf/_frozen_experiment_v1.py'


def require_active(backend):
    if backend not in ACTIVE_BACKENDS:
        raise ValueError(f'{backend} is retired by user instruction; no new experiments are permitted')


def check_training_hashes(record):
    hashes=record.get('source_hashes',{});errors=[]
    for relative in ['density_cf/context.py','density_cf/birch.py','density_cf/hdb.py','density_cf/experiment.py']:
        path=ROOT/relative;expected=hashes.get(relative)
        if relative=='density_cf/experiment.py':
            # Keep exact historical bytes rather than rewriting old hashes.
            path=ROOT/FROZEN_ENGINE
            expected=hashes.get(FROZEN_ENGINE,expected)
        if expected!=hashlib.sha256(path.read_bytes()).hexdigest():errors.append(relative)
    return errors
