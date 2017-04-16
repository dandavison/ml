from functools import partial

from scipy.stats import describe


describe = partial(describe, axis=None)
