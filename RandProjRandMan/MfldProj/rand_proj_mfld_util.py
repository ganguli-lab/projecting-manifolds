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
from typing import List, Dict, Optional
import numpy as np

from ..RandCurve import gauss_surf as gs
from ..iter_tricks import dcontext


def mat_field_evals(mat_field: np.ndarray) -> np.ndarray:
    """eigenvalues of stacks of symmetric matrices

    Parameters
    ----------
    mat_field : np.ndarray, (Lx,Ly,...,N,N)
        2nd rank tensor field.

    Returns
    -------
    evals : np.ndarray, (Lx,Ly,...,N)
        eigenvalues, descending order
    """
    if mat_field.shape[-1] == 1:
        return mat_field.squeeze(-1)
    elif mat_field.shape[-1] == 2:
        return np.stack(gs.mat_field_evals(mat_field), axis=-1)
    return np.linalg.eigvalsh(mat_field)


def endval(param_dict: Dict[str, np.ndarray],
           param: str) -> Dict[str, np.ndarray]:
    """Replace elements of array in dictionary with its last element.
    """
    new_param_dict = param_dict.copy()
    new_param_dict[param] = param_dict[param][-1:]
    return new_param_dict

# =============================================================================
# %%* generate manifold / projection
# =============================================================================


def make_basis(num_samp: int,
               ambient_dim: int,
               proj_dim: int) -> np.ndarray:  # basis for random space
    """
    Generate orthonormal basis for projection subspace

    Parameters
    ----------
    num_samp
        S, number of samples of subspaces
    ambient_dim
        N, dimensionality of ambient space
    proj_dim
        M, dimensionality of subspace

    Returns
    -------
    U
        bases of subspaces for each sample projection, (S,N,M)
    """
    spaces = np.random.randn(num_samp, ambient_dim, proj_dim)
#    return np.array([np.linalg.qr(u)[0] for u in U])
    proj = np.empty((num_samp, ambient_dim, proj_dim))
    for i, space in enumerate(spaces):
        # orthogonalise with Gram-Schmidt
        proj[i] = np.linalg.qr(space)[0]
    return proj


def project_mfld(mfld: np.ndarray,
                 gmap: np.ndarray,
                 proj_dim: int,
                 num_samp: int) -> (np.ndarray, List[np.ndarray]):
    """Project manifold and gauss_map

    Parameters
    ----------
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
    with dcontext('Projections'):
        # sample projectors, (S,N,max(M))
        projs = make_basis(num_samp, mfld.shape[-1], proj_dim)
    with dcontext('Projecting'):
        # projected manifold for each sampled proj, (S,Lx*Ly...,max(M))
        proj_mflds = mfld @ projs
        # gauss map of projected mfold for each proj, (#K,)(S,L,K,max(M))
        pgmap = [gmap[:, :k+1] @ projs[:, None] for k in range(gmap.shape[1])]
    return proj_mflds, pgmap


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
