"""Skip test modules whose OPTIONAL heavy-solver extras are not installed.

The core platform — and the default backend install (``.[dev,ingest,fusion]``) — does NOT
depend on the rigorous-forward (M3), inversion (M9), or geomodel (M8) solver libraries.
Those are heavy, optional extras (doc 10 is explicitly later-phase/non-blocking; doc 08
``executionMode`` worker/container). So their tests are skipped at collection time when the
extra is absent, keeping ``make test`` green on the default install. Install the solvers
(``make install-backend-full``) to exercise them.
"""

from __future__ import annotations

import importlib.util


def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


# test file -> required modules (the file is skipped if ANY is missing)
_REQUIRES: dict[str, list[str]] = {
    # M3 rigorous (T1) forward models
    "test_t1_gravity.py": ["harmonica"],
    "test_t1_mt.py": ["empymod"],
    "test_t1_seismic.py": ["pykonal"],
    "test_m3_exit.py": ["harmonica", "empymod", "pykonal"],
    # M9 inversion (geosim.inversion imports `discretize` at module load)
    "test_inversion_engine.py": ["discretize"],
    "test_inversion_gravity.py": ["discretize", "simpeg"],
    "test_inversion_ert.py": ["discretize", "pygimli"],
    "test_inversion_cooperative.py": ["discretize"],
    "test_m9_exit.py": ["discretize", "simpeg", "pygimli"],
    # M8 implicit geomodel
    "test_geomodel.py": ["gempy"],
    "test_m8_exit.py": ["gempy"],
}

collect_ignore = [f for f, mods in _REQUIRES.items() if not all(_have(m) for m in mods)]
