# -*- coding: utf-8 -*-
# =============================================================================
# Created on Mon Jun 19 13:29:22 2017
#
# @author: Subhy based on Peiran's Matlab code
#
# Module: rand_proj_mfld_mem
# =============================================================================
"""
Calculation of distribution of maximum distortion of Gaussian random manifolds
under random projections, low memory version

Functions
=========
region_inds_list
    indices of points and pairs of points on mfld corresponding to the central
    region of the manifold
distortion_m
    Maximum distortion of all chords between points on the manifold,
    sampling projectors, for each V, M
"""
from typing import Sequence, Tuple, List, Mapping
from numbers import Real
from math import floor
import numpy as np

from ..iter_tricks import dbatch, denumerate, rdenumerate, dcontext
from ..RandCurve import gauss_mfld as gm
from . import distratio as dr

Nind = np.ndarray  # Iterable[int]  # Set[int]
Pind = np.ndarray  # Iterable[Tuple[int, int]]  # Set[Tuple[int, int]]
Inds = Tuple[Nind, Pind]

# =============================================================================
# %%* generate projection
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


def region_indices(shape: Sequence[int],
                   mfld_frac: float,
                   random: bool = False) -> List[np.ndarray]:
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


def region_inds_list(shape: Sequence[int],
                     mfld_fs: Sequence[float]) -> List[List[Inds]]:
    """
    List of index sets for different sized regions, each index set being the
    indices of condensed matrix returned by pdist corresponding to the
    central region of the manifold

    Parameters
    ----------
    shape
        tuple of nujmber of points along each dimension (max(K),)
    mfld_fs
        list of fractions of manifold to keep (#(V))

    Returns
    -------
    region_inds
        list of lists of tuples of arrays containing indices of: new & previous
        points in K-d subregions (#(V),)(#(K),)(2,), each element an array of
        indices of shape ((fL)^K - #(prev),) or (#(prev),), where:
        #(prev) = (fL)^K-1 + (f'L)^K - (f'L)^K-1
    """
    # new indices, & those seen before, for each f, K
    region_inds = []
    # all indices for previous f, for all K
    prev_fs = [np.array([], int) for i in range(len(shape))]
    # loop over f
    for frac in mfld_fs:
        # all indices, for this f, for all K
        all_inds = region_indices(shape, frac)
        # arrays to store new & previous for this f, all K
        ind_arrays = []
        # all indices, for new f, for previous K
        prev_K = np.array([], int)
        # loop over K
        for aind, prev_f in zip(all_inds, prev_fs):
            # indices seen before this f & K
            pind = np.union1d(prev_f, prev_K)
            # remove previous f & K to get new indices
            nind = np.setdiff1d(aind, pind, assume_unique=True)
            # store new & previous for this K
            ind_arrays.append((nind, pind))
            # update all indices for this f, previous K (next K iteration)
            prev_K = aind
        # store new & previous for this f, all K
        region_inds.append(ind_arrays)
        # update all indices for previous f, all K (next f iteration)
        prev_fs = all_inds
    return region_inds


# =============================================================================
# %%* distortion calculations
# =============================================================================


def distortion_gmap(proj_gmap: Sequence[np.ndarray], N: int) -> np.ndarray:
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


def distortion(vecs: np.ndarray, pvecs: np.ndarray, inds: Inds) -> np.ndarray:
    """Distortion of a chord

    Parameters
    ----------
    vecs : np.ndarray (L,N,)
        points in the manifold
    pvecs : np.ndarray (L,M)
        corresponding points in the projected manifold
    inds : Tuple(np.ndarray[int], np.ndarray[int])
        tuples of arrays containing indices of: new & previous points in
        subregions (2,), each element an array of indices of shape
        ((fL)^K - #(prev),) or (#(prev),),
        where: #(prev) = (fL)^K-1 + (f'L)^K - (f'L)^K-1

    Returns
    -------
    distortion : np.ndarray (S,)
        maximum distortion of chords
    """
    scale = np.sqrt(vecs.shape[-1] / pvecs.shape[-1])
    distn = 0.
    if len(inds[0]) > 0:
        lratio = dr.pdist_ratio(vecs[inds[0]], pvecs[inds[0]])
        distn = np.maximum(distn, np.abs(scale * lratio - 1.).max())
        if len(inds[1]) > 0:
                lratio = dr.cdist_ratio(vecs[inds[0]], vecs[inds[1]],
                                        pvecs[inds[0]], pvecs[inds[1]])
                distn = np.maximum(distn, np.abs(scale * lratio - 1.).max())
    return distn


def distortion_v(mfld: np.ndarray,
                 proj_mflds: np.ndarray,
                 proj_gmap: Sequence[np.ndarray],
                 region_inds: Sequence[Sequence[Inds]]) -> np.ndarray:
    """
    Max distortion of all tangent vectors and chords between points in various
    regions manifold, for all V

    Parameters
    ----------
    mfld[st...,i]
        phi_i(x[s],y[t],...),  (L,N),
        matrix of points on manifold as row vectors,
        i.e. flattened over intrinsic location indices
    proj_mfld[q,st...,i]
        phi_i(x[s],y[t],...),  (S,L,M),
        projected manifolds, first index is sample #
    proj_gmap[k][q,st...,A,i]
        list of e_A^i(x[s],y[t],...),  (#(K),)(S,L,K,M),
        list members: gauss map of projected manifolds, 1sts index is sample #
    region_inds
        list of lists of tuples of arrays containing indices of: new & previous
        points in K-d subregions (#(V),)(#(K),)(2,), each element an array of
        indices of shape ((fL)^K - #(prev),) or (#(prev),), where:
        #(prev) = (fL)^K-1 + (f'L)^K - (f'L)^K-1

    Returns
    -------
    epsilon = max distortion of all chords (#(K),#(V),S)
    """
    # tangent space distortions, (#(K),)(S,L)
    gdistn = distortion_gmap(proj_gmap, mfld.shape[-1])

    distn = np.empty((len(region_inds[0]),
                      len(region_inds),
                      proj_mflds.shape[0]))  # (#(K),#(V),S)

    for v, inds in denumerate('Vol', region_inds):
        for k, gdn, pts in denumerate('K', gdistn, inds):
            distn[k, v] = gdn[:, pts[0]].max(axis=-1)  # (S,)

            for s, pmfld in denumerate('S', proj_mflds):
                    np.maximum(distn[k, v, s], distortion(mfld, pmfld, pts),
                               out=distn[k, v, s:s+1])

    # because each entry in region_inds  only contains new chords
    np.maximum.accumulate(distn, axis=0, out=distn)  # (#(K),#(V),S)
    np.maximum.accumulate(distn, axis=1, out=distn)  # (#(K),#(V),S)

    return distn


def distortion_m(mfld: np.ndarray,
                 gmap: np.ndarray,
                 proj_dims: np.ndarray,
                 uni_opts: Mapping[str, Real],
                 region_inds: Sequence[Sequence[Inds]]) -> np.ndarray:
    """
    Maximum distortion of all chords between points on the manifold,
    sampling projectors, for each V, M

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
    proj_dims
        ndarray of M's, dimensionalities of projected space (#(M),)
    uni_opts
            dict of scalar options, used for all parameter values, with fields:
        num_samp
            number of samples of distortion for empirical distribution
        batch
            sampled projections are processed in batches of this length.
            The different batches are looped over (mem version).
        chunk
            chords are processed (vectorised) in chunks of this length.
            The different chunks are looped over (mem version).
    region_inds
        list of lists of tuples of arrays containing indices of: new & previous
        points in K-d subregions (#(V),)(#(K),)(2,), each element an array of
        indices of shape ((fL)^K - #(prev),) or (#(prev),), where:
        #(prev) = (fL)^K-1 + (f'L)^K - (f'L)^K-1

    Returns
    -------
    epsilon = max distortion of chords for each (#(K),#(M),#(V),S)
    """
    # preallocate output. (#(K),#(M),#(V),S)
    distn = np.empty((len(region_inds[0]), len(proj_dims),
                      len(region_inds), uni_opts['samples']))

    batch = uni_opts['batch']
    for s in dbatch('Sample', 0, uni_opts['samples'], batch):
        # projected manifold for each sampled proj, (S,Lx*Ly...,max(M))
        # gauss map of projected mfold for each proj, (#K,)(S,L,K,max(M))
        pmflds, pgmap = project_mfld(mfld, gmap, proj_dims[-1], batch)

        # loop over M
        for i, M in rdenumerate('M', proj_dims):
            # distortions of all chords in (K-dim slice of) manifold
            distn[:, i, :, s] = distortion_v(mfld, pmflds[..., :M],
                                             [pgm[..., :M] for pgm in pgmap],
                                             region_inds)
    return distn


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
