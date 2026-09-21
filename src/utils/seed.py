"""
Global seeding for reproducibility (CLAUDE.md section 9: seed = 42 everywhere).

Every training/eval entry point calls set_global_seed() before doing anything
else, so that a rerun with the same config produces the same numbers.
"""

import os
import random

import numpy as np

SEED = 42


def set_global_seed(seed: int = SEED, deterministic: bool = True):
    """Seed random, numpy, and torch (CPU + CUDA).

    deterministic=True also forces cuDNN into deterministic mode. That costs
    some speed but means two runs of the same config give identical results --
    which is what makes the with/without ablations in Phase 5 trustworthy
    (a difference in the metric is then caused by the ablated variable, not
    by nondeterministic kernel scheduling).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Imported lazily so that data-only scripts can use this module without torch installed.
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
