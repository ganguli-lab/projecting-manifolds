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
Matrix = nb.float64[:, :]

# =============================================================================
# functions
# =============================================================================


@nb.jit(nb.float64(nb.float64[:]), cache=True, nopython=True)
def _norm(vec: np.ndarray) -> float:
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
    return np.sum(vec**2)


@nb.jit(FloatPair(Matrix, Matrix), cache=True, nopython=True)
def pdist_ratio(X: np.ndarray, P: np.ndarray) -> (float, float):
    """Max and min ratio of pair-wise distances squared beween corresponding
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
    for i in range(X.shape[0]):
        for j in range(i):
            ratio = norm(P[i] - P[j]) / norm(X[i] - X[j])
            if ratio > drmax:
                drmax = ratio
            if ratio < drmin:
                drmin = ratio
    return drmax, drmin


@nb.jit(FloatPair(Matrix, Matrix, Matrix, Matrix), cache=True, nopython=True)
def cdist_ratio(XA: np.ndarray, XB: np.ndarray,
                PA: np.ndarray, PB: np.ndarray) -> (float, float):
    """Max and min ratio of cross-wise distances squared beween corresponding
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
    for i in range(XA.shape[0]):
        for j in range(XB.shape[0]):
            ratio = norm(PA[i] - PB[j]) / norm(XA[i] - XB[j])
            if ratio > drmax:
                drmax = ratio
            if ratio < drmin:
                drmin = ratio
    return drmax, drmin
