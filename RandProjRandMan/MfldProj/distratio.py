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

try:
    from _distratio import pdist_ratio, cdist_ratio
except ImportError:
    try:
        from _distratio_jit import pdist_ratio, cdist_ratio
    except ImportError:
        from _distratio_scp import pdist_ratio, cdist_ratio
