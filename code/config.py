import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.environ.get('ELLPP_DIR', os.path.join(ROOT, 'data', 'ellipticpp'))
CACHE = os.environ.get('ELLPP_CACHE', os.path.join(ROOT, 'cache'))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
TRAIN_END = 29
VAL_END = 34
LAST_STEP = 49
SEEDS = list(range(10))
for p in (CACHE, RES, FIG):
    os.makedirs(p, exist_ok=True)
