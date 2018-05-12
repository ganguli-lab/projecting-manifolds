# -*- coding: utf-8 -*-
"""
Created on Fri May 11 18:28:26 2018

@author: Subhy

distratio
=========
Max and min ratios of cross/pair-wise distances squared.
"""
import numpy as np
from numpy.linalg import norm
import numba as nb
FloatPair = nb.types.UniTuple(nb.float64, 2)

# =============================================================================
# functions
# =============================================================================


@nb.jit(FloatPair(nb.float64[:, :], nb.float64[:, :]), cache=True)
def pdist_ratio(X: np.ndarray, P: np.ndarray) -> (float, float):
    """
    Max and min ratio of pair-wise distances squared beween corresponding
    pairs of points in two sets.

    Parameters
    -----------
    X: ndarray
        Set of points between which we compute pairwise distances for the
        denominator. Each point is a row.
    P: ndarray
        Set of points between which we compute pairwise distances for the
        numerator.

    Returns
    -------
    drmax: double
        Maximum ratio of distances squared.
    drmin: double
        Minimum ratio of distances squared.
    """
    drmax = 0.
    drmin = np.Inf
    for i, (xa, pa) in enumerate(zip(X, P)):
        for xb, pb in zip(X[:i], P[:i]):
            ratio = norm(pa - pb) / norm(xa - xb)
            if ratio > drmax:
                drmax = ratio
            if ratio < drmin:
                drmin = ratio
    return drmax, drmin


@nb.jit(FloatPair(nb.float64[:, :], nb.float64[:, :],
                  nb.float64[:, :], nb.float64[:, :]), cache=True)
def cdist_ratio(XA: np.ndarray, XB: np.ndarray,
                PA: np.ndarray, PB: np.ndarray) -> (float, float):
    """
    Max and min ratio of cross-wise distances squared beween corresponding
    pairs of points in two groups of two sets.

    Parameters
    -----------
    XA: ndarray
        Set of points *from* which we compute pairwise distances for the
        denominator. Each point is a row.
    XB: ndarray
        Set of points *to* which we compute pairwise distances for the
        denominator.
    PA: ndarray
        Set of points *from* which we compute pairwise distances for the
        numerator.
    PB: ndarray
        Set of points *to* which we compute pairwise distances for the
        numerator.

    Returns
    -------
    drmax: double
        Maximum ratio of distances squared.
    drmin: double
        Minimum ratio of distances squared.
    """
    drmax = 0.
    drmin = np.Inf
    for xa, pa in zip(XA, PA):
        for xb, pb in zip(XB, PB):
            ratio = norm(pa - pb) / norm(xa - xb)
            if ratio > drmax:
                drmax = ratio
            if ratio < drmin:
                drmin = ratio
    return drmax, drmin
