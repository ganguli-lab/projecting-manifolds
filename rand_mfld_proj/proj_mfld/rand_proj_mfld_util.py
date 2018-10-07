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
from typing import Dict, Optional, Sequence, List, Tuple
from numbers import Real
from math import floor
import numpy as np
from ..proj import intra_cell as ic
from ..mfld import gauss_mfld as gm
from ..iter_tricks import dcontext
from ..larray import larray


def endval(param_dict: Dict[str, larray],
           param: str) -> Dict[str, larray]:
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


def pairs(vec: larray, other: Optional[larray] = None) -> larray:
    """pairs of elements

    Parameters
    ----------
    vec : larray, (M,)
        Vector of elements for first member of pair.
    other : Optional[larray], (N,)
        Vector of elements for second member of pair.
        If None (default), `other` = `vec`, and only distinct unordered pairs
        are returned.

    Returns
    -------
    pairs : larray, (2,MN) or (2,M(M-1)/2)
        Pairs of elements from `vec` and `other`, or both from `vec`.
    """
    if other is None:
        pairs = np.stack(np.broadcast_arrays(*np.ix_(vec, vec)))
        ind1, ind2 = np.tril_indices(vec.size, -1)
        return pairs[:, ind1, ind2]
    return np.stack(np.broadcast_arrays(*np.ix_(vec, other))).reshape((2, -1))


def region_indices(shape: Sequence[int],
                   mfld_frac: float,
                   random: bool = False) -> List[larray]:
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


def project_mfld(mfld_bundle: Tuple[larray, larray],
                 proj_dim: int,
                 num_samp: int) -> (larray, List[larray]):
    """Project manifold and gauss_map

    Parameters
    ----------
    mfld-bundle
        Tuple: (mfld, gmap)
    mfld[st...,i]
        phi_i(x[s],y[t],...),  (L,N),
        matrix of points on manifold as row vectors,
        i.e. flattened over intrinsic location indices
    gmap[st...,A,i]
        e_A^i(x[s,t...])., (L,K,N),
        orthonormal basis for tangent space,
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
    K, N = mfld_bundle[1].shape[-2:]
    with dcontext('Projections'):
        # sample projectors, (S,N,max(M))
        projs = ic.make_basis(N, proj_dim, num_samp)
    with dcontext('Projecting'):
        # projected manifold for each sampled proj, (S,Lx*Ly...,max(M))
        proj_mflds = mfld_bundle[0] @ projs
        # projected manifold for each sampled proj, (S,Lx*Ly...,K,max(M))
        proj_gmap = mfld_bundle[1] @ projs[:, None]
        # gauss map of projected mfold for each proj, (#K,)(S,L,K,max(M))
        pgmap = [proj_gmap[:, :, :k+1] for k in range(K)]
    return proj_mflds, pgmap


# =============================================================================
# %%* distortion calculations
# =============================================================================


def distortion_gmap(proj_gmap: Sequence[larray], N: int) -> larray:
    """
    Max distortion of all tangent vectors

    Parameters
    ----------
    proj_gmap[k][q,st...,A,i]
        list of e_A^i(x[s],y[t],...),  (#(K),)(S,L,K,M),
        list members: gauss map of projected manifolds, 1sts index is sample #

    Returns
    -------
    epsilon = max distortion of all chords (#(K),#(V),S)
    """
    M = proj_gmap[-1].shape[-1]

    # tangent space/projection angles, (#(K),)(S,L,K)
    cossq = [gm.mat_field_svals(v) for v in proj_gmap]
    # tangent space distortions, (#(K),)(S,L)
    gdistn = [np.abs(np.sqrt(c * N / M) - 1).max(axis=-1) for c in cossq]
    return gdistn
