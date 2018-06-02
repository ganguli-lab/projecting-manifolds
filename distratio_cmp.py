# -*- coding: utf-8 -*-
"""
Created on Fri May 11 18:28:26 2018

@author: Subhy

distratio
=========
Max and min ratios of cross/pair-wise distances squared.
"""
import os.path as osp
import numpy as np
from numpy.linalg import norm
import numba as nb

# =============================================================================
# Compiler class
# =============================================================================

# hack for windows: tempfile locks temporary files so cl.exe can't compile them
# Numba mistakes this for lack of a compiler.
import numba.pycc.platform as nbplat
nbplat._external_compiler_ok = True

dr_mod = nb.pycc.CC('distratio')
dr_mod.output_dir = osp.join(dr_mod.output_dir, 'RandProjRandMan', 'MfldProj')

# =============================================================================
# functions
# =============================================================================


# =============================================================================
# @dr_mod.export('norm', 'f8(f8[:])')
# def norm(vec: np.ndarray) -> float:
#     """Squared norm of a vector.
#
#     Parameters
#     -----------
#     vec: ndarray
#         A vector.
#
#     Returns
#     -------
#     nrm: double
#         Euclidean 2-norm squared.
#     """
#     return np.sum(vec**2)
# =============================================================================

@dr_mod.export('pdist_ratio', 'f8[:](f8[:,:],f8[:,:])')
def pdist_ratio(X: np.ndarray, P: np.ndarray) -> (float, float):
    """Max and min ratio of pair-wise distances beween corresponding pairs of
    points in two sets.

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
    dr_range: ndarray
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


@dr_mod.export('cdist_ratio', 'f8[:](f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
def cdist_ratio(XA: np.ndarray, XB: np.ndarray,
                PA: np.ndarray, PB: np.ndarray) -> (float, float):
    """Max and min ratio of cross-wise distances beween corresponding pairs of
    points in two groups of two sets.

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
    dr_range: ndarray
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


# =============================================================================
# Compile it
# =============================================================================


if __name__ == "__main__":
    dr_mod.compile()
