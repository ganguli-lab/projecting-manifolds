# -*- coding: utf-8 -*-
"""
Created on Fri May 11 18:28:26 2018

@author: Subhy

distratio
=========
Max and min ratios of cross/pair-wise distances squared.

Functions
=========
pdist_ratio
    Max and min ratio of pair-wise distances squared beween corresponding
    pairs of points in two sets.
cdist_ratio
    Max and min ratio of cross-wise distances squared beween corresponding
    pairs of points in two groups of two sets.
"""
from functools import wraps
import numpy as np

try:
    from ._distratio import pdist_ratio as _pdist_ratio
    from ._distratio import cdist_ratio as _cdist_ratio

    @wraps(_pdist_ratio)
    def pdist_ratio(*args, **kwds):
        return np.sqrt(np.stack(_pdist_ratio(*args, **kwds), axis=-1))

    @wraps(_cdist_ratio)
    def cdist_ratio(*args, **kwds):
        return np.sqrt(np.stack(_cdist_ratio(*args, **kwds), axis=-1))
except ImportError:
    from ._distratio_scp import pdist_ratio, cdist_ratio
#     try:
#         from ._distratio_jit import pdist_ratio, cdist_ratio
#     except ImportError:
