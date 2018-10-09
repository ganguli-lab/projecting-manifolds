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
from typing import Dict, Optional, Sequence, List
from numbers import Real
from math import floor
import numpy as np
from numpy import ndarray as array

from ..proj import intra_cell as ic
from ..mfld import gauss_mfld as gm
from ..iter_tricks import dcontext


def endval(param_dict: Dict[str, array],
           param: str) -> Dict[str, array]:
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


def pairs(vec: array, other: Optional[array] = None) -> array:
    """pairs of elements

    Parameters
    ----------
    vec : array, (M,)
        Vector of elements for first member of pair.
    other : Optional[array], (N,)
        Vector of elements for second member of pair.
        If None (default), `other` = `vec`, and only distinct unordered pairs
        are returned.

    Returns
    -------
    pairs : array, (2,MN) or (2,M(M-1)/2)
        Pairs of elements from `vec` and `other`, or both from `vec`.
    """
    if other is None:
        pairs = np.stack(np.broadcast_arrays(*np.ix_(vec, vec)))
        ind1, ind2 = np.tril_indices(vec.size, -1)
        return pairs[:, ind1, ind2]
    return np.stack(np.broadcast_arrays(*np.ix_(vec, other))).reshape((2, -1))


def region_indices(shape: Sequence[int],
                   mfld_frac: float,
                   random: bool = False) -> List[array]:
    """
    Indices of points corresponding to the central region of the manifold.
    Smaller `mfld_frac` is guaranteed to return a subset of larger `mfld_frac`.

    Parameters
    ----------
    shape
        tuple of number of points along each dimension (max(K),)
    mfld_frac
        fraction of manifold to keep

    Returns
    -------
    lin_inds
        set of indices of points on manifold, restricted to
        K-d central region, (#(K),)((fL)^K)
    """
    ranges = ()
    midranges = ()
    for siz in shape:
        # how many elements to remove?
        remove = floor((1. - mfld_frac)*siz)
        if random:
            # how many of those to remove from start?
            removestart = np.random.randint(remove + 1)
            # which point to use for lower K?
            mid = np.random.randint(siz)
        else:
            # how many of those to remove from start?
            removestart = remove // 2
            # which point to use for lower K?
            mid = siz // 2
        # slice for region in eack dimension
        ranges += (np.arange(removestart, siz + removestart - remove),)
        # slice for point in each dimension, needed for lower K
        midranges += (np.array([mid]),)
    all_ranges = [ranges[:k] + midranges[k:] for k in range(1, len(shape) + 1)]
    # indices of region in after ravel
    lin_inds = [np.ravel_multi_index(np.ix_(*range_k), shape).ravel()
                for range_k in all_ranges]
    return lin_inds


# =============================================================================
# %%* generate projection
# =============================================================================


def project_mfld(mfld: gm.SubmanifoldFTbundle,
                 proj_dim: int,
                 num_samp: int) -> gm.SubmanifoldFTbundle:
    """Project manifold and gauss_map

    Parameters
    ----------
    mfld: SubmanifoldFTbundle
        mfld[s,t,...,i]
            = phi_i(x[s],y[t],...), (Lx,Ly,...,N)
            Embedding functions of random surface
        gmap[s,t,i,A]
            = e_A^i(x[s], y[t]).
            orthonormal basis for tangent space, (Lx,Ly,N,K)
            e_(A=0)^i must be parallel to d(phi^i)/dx^(a=0)
    proj_dim
        M, dimensionalities of projected space (#(M),)
    num_samp
        S, # samples of projectors for empirical distribution

    Returns
    -------
    proj_mfld[q,st...,i]
        phi_i(x[s],y[t],...),  (S,L,M),
        projected manifolds, first index is sample #
    proj_gmap[k][q,st...,A,i]
        tuple of e_A^i(x[s],y[t],...),  (K,)(S,L,K,M),
        tuple members: gauss map of projected manifolds, 1sts index is sample #
    """
    with dcontext('Projections'):
        # sample projectors, (S,N,M)
        projs = ic.make_basis(num_samp, mfld.ambient, proj_dim)
    with dcontext('Projecting'):
        proj_mflds = gm.SubmanifoldFTbundle()
        proj_mflds.ambient = proj_dim
        proj_mflds.intrinsic = mfld.intrinsic
        proj_mflds.shape = (num_samp,) + mfld.shape
        # projected manifold for each sampled proj, (S,Lx*Ly...,M)
        proj_mflds.mfld = mfld.mfld @ projs
        # gauss map of projected mfold for each proj, (S,L,K,M)
        proj_mflds.gmap = mfld.gmap @ projs[:, None]
    return proj_mflds


# =============================================================================
# %%* distortion calculations
# =============================================================================


def distortion_gmap(proj_mfld: gm.SubmanifoldFTbundle, N: int) -> array:
    """
    Max distortion of all tangent vectors

    Parameters
    ----------
    proj_mfld: SubmanifoldFTbundle
        mfld[q,st...,i]
            phi_i(x[s],y[t],...),  (S,L,M),
            projected manifolds, first index is sample #
        gmap[q,st...,A,i]
            e_A^i(x[s],y[t],...),  (S,L,K,M),
            gauss map of projected manifolds, 1sts index is sample #
    ambient_dim
        N, dimensionality of ambient space

    Returns
    -------
    epsilon = max distortion of all chords (#(K),#(V),S)
    """
    M = proj_mfld.ambient
    K = proj_mfld.intrinsic

    # gauss map of projected mfold for each K, (#K,)(S,L,K,max(M))
    proj_gmap = [proj_mfld.gmap[:, :, :k+1] for k in range(K)]
    # tangent space/projection angles, (#(K),)(S,L,K)
    cossq = [gm.mat_field_svals(v) for v in proj_gmap]
    # tangent space distortions, (#(K),)(S,L)
    gdistn = [np.abs(np.sqrt(c * N / M) - 1).max(axis=-1) for c in cossq]
    return gdistn
