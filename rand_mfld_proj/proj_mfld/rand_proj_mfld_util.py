# -*- coding: utf-8 -*-
# =============================================================================
# Created on Fri Jan 12 17:24:12 2018
#
# @author: Subhy
#
# Module: rand_proj_mfld_util
# =============================================================================
"""
Utilities for calculation of distribution of maximum distortion of Gaussian
random manifolds under random projections, low memory version
"""
from typing import Dict, Optional, Sequence
from numbers import Real
import numpy as np


def endval(param_dict: Dict[str, np.ndarray],
           param: str) -> Dict[str, np.ndarray]:
    """Replace elements of array in dictionary with its last element.
    """
    new_param_dict = param_dict.copy()
    new_param_dict[param] = param_dict[param][-1:]
    return new_param_dict


def gmean(data: Sequence[Real]) -> float:
    """Geometric mean of a sequence of numbers
    """
    return np.prod(data)**(1./len(data))


# =============================================================================
# %%* region indexing
# =============================================================================


def pairs(vec: np.ndarray, other: Optional[np.ndarray] = None) -> np.ndarray:
    """pairs of elements

    Parameters
    ----------
    vec : np.ndarray, (M,)
        Vector of elements for first member of pair.
    other : Optional[np.ndarray], (N,)
        Vector of elements for second member of pair.
        If None (default), `other` = `vec`, and only distinct unordered pairs
        are returned.

    Returns
    -------
    pairs : np.ndarray, (2,MN) or (2,M(M-1)/2)
        Pairs of elements from `vec` and `other`, or both from `vec`.
    """
    if other is None:
        pairs = np.stack(np.broadcast_arrays(*np.ix_(vec, vec)))
        ind1, ind2 = np.tril_indices(vec.size, -1)
        return pairs[:, ind1, ind2]
    return np.stack(np.broadcast_arrays(*np.ix_(vec, other))).reshape((2, -1))
