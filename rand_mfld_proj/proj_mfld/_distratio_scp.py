# -*- coding: utf-8 -*-
"""
Created on Fri May 11 18:28:26 2018

@author: Subhy

distratio
=========
Max and min ratios of cross/pair-wise distances squared.
"""
import numpy as np
from scipy.spatial.distance import pdist, cdist

# =============================================================================
# functions
# =============================================================================


def pdist_ratio(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Max and min ratio of pair-wise distances squared beween corresponding
    pairs of points in two sets.

    Min,max of norm(P[i] - P[j]) / norm(X[i] - X[j]).

    Parameters
    -----------
    X: ndarray (S,N)
        Set of points between which we compute pairwise distances for the
        denominator. Each point is a row.
    P: ndarray (S,M)
        Set of points between which we compute pairwise distances for the
        numerator.

    Returns
    -------
    dr_range: ndarray (2,)
        [Minimum, Maximum] ratio of distances.
    """
    r = pdist(P) / pdist(X)
    return np.array([r.min(), r.max()])


def cdist_ratio(XA: np.ndarray, XB: np.ndarray,
                PA: np.ndarray, PB: np.ndarray) -> np.ndarray:
    """Max and min ratio of cross-wise distances squared beween corresponding
    pairs of points in two groups of two sets.

    Min,max of norm(PA[i] - PB[j]) / norm(XA[i] - XB[j]).

    Parameters
    -----------
    XA: ndarray (S,N)
        Set of points *from* which we compute pairwise distances for the
        denominator. Each point is a row.
    XB: ndarray (T,N)
        Set of points *to* which we compute pairwise distances for the
        denominator.
    PA: ndarray (S,M)
        Set of points *from* which we compute pairwise distances for the
        numerator.
    PB: ndarray (T,M)
        Set of points *to* which we compute pairwise distances for the
        numerator.

    Returns
    -------
    dr_range: ndarray (2,)
        [Minimum, Maximum] ratio of distances.
    """
    r = cdist(PA, PB) / cdist(XA, XB)
    return np.array([r.min(), r.max()])
