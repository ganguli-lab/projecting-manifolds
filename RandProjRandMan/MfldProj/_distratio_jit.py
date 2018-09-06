# -*- coding: utf-8 -*-
"""
Created on Fri May 11 18:28:26 2018

@author: Subhy

distratio
=========
Max and min ratios of cross/pair-wise distances squared.
"""
import numpy as np
# from numpy.linalg import norm
from numba import jit, f8

# =============================================================================
# functions
# =============================================================================


@jit(f8(f8[:]), cache=True, nopython=True)
def norm(vec: np.ndarray) -> float:
    """Squared norm of a vector.

    Parameters
    -----------
    vec: ndarray
        A vector.

    Returns
    -------
    nrm: double
        Euclidean 2-norm squared.
    """
    s = 0.
    for i in range(vec.shape[0]):
        s += vec[i]**2
    return s


@jit(f8[:](f8[:, :], f8[:, :]), cache=True, nopython=True)
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
    drmax = 0.
    drmin = np.Inf
    for i in range(X.shape[0]):
        for j in range(i):
            ratio = norm(P[i] - P[j]) / norm(X[i] - X[j])
            if ratio > drmax:
                drmax = ratio
            if ratio < drmin:
                drmin = ratio
    return np.array([drmin, drmax])


@jit(f8[:](f8[:, :], f8[:, :], f8[:, :], f8[:, :]), cache=True, nopython=True)
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
    drmax = 0.
    drmin = np.Inf
    for i in range(XA.shape[0]):
        for j in range(XB.shape[0]):
            ratio = norm(PA[i] - PB[j]) / norm(XA[i] - XB[j])
            if ratio > drmax:
                drmax = ratio
            if ratio < drmin:
                drmin = ratio
    return np.array([drmin, drmax])
