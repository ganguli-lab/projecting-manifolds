# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:29:22 2017

@author: Subhy based on Peiran's Matlab code

Compute distribution of maximum distortion of Gaussian random manifolds under
random projections

Functions
=========
get_num_numeric
    calculate numeric quantities when varying ambient dimensions
get_vol_numeric
    calculate numeric quantities when varying size of manifold
get_all_numeric
    calculate all numeric quantities
default_options
    default options for long numerics for paper
quick_options
    default options for quick numerics for demo
make_and_save
    generate data and save npz file
"""
from typing import Sequence, Tuple, List, Mapping
from numbers import Real
from math import floor
import numpy as np

from ..iter_tricks import dbatch, denumerate
from . import rand_proj_mfld_util as ru

Lind = np.ndarray  # Iterable[int]  # Set[int]
Pind = np.ndarray  # Iterable[Tuple[int, int]]  # Set[Tuple[int, int]]
Inds = Tuple[Sequence[Lind], Sequence[Pind]]

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
        tuple of number of points along each dimension (K,)
    mfld_frac
        fraction of manifold to keep

    Returns
    -------
    lin_ind1, lin_ind2
        set of indices of points on manifold, restricted to
        (1d slice of central region, 2d central region), (#K,)(fL)
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
                     mfld_fs: Sequence[float]) -> List[Inds]:
    """
    List of index sets for different sized regions, each index set being the
    indices of condensed matrix returned by pdist corresponding to the
    central region of the manifold

    Parameters
    ----------
    shape
        tuple of nujmber of points along each dimension (K,)
    mfld_fs
        list of fractions of manifold to keep (#(V))

    Returns
    -------
    region_inds
        list of tuples of arrays containing indices of: new points & new pairs
        in 1d and 2d subregions (#(V),2,#(K)),
        each element an array of indices (fL,) or (fL(fL-1)/2,)
    """
    region_inds = []
    sofars = [np.array([], int) for i in range(len(shape))]
    for frac in mfld_fs:
        # indices for regions we keep
        linds = region_indices(shape, frac)
        # arrays to store them
        lind_arrays = []
        pind_arrays = []
        # new entries
        for lind, sofar in zip(linds, sofars):
            lind = np.setdiff1d(lind, sofar, assume_unique=True)
            # pairs: new-new + new-old
            pind = np.concatenate((ru.pairs(lind).T,
                                   ru.pairs(lind, sofar).T))
            # update set of old entries
            sofar = np.union1d(sofar, lind)
            # add to lists
            lind_arrays.append(lind)
            pind_arrays.append(np.array(pind))
        # store
        region_inds.append((lind_arrays, pind_arrays))
#        region_inds.append(region_squareform_inds(shape, frac))
    return region_inds


# =============================================================================
# %%* distortion calculations
# =============================================================================


def distortion_vec(vec: np.ndarray, proj_vec: np.ndarray) -> np.ndarray:
    """Distortion of a chord

    Parameters
    ----------
    vec : np.ndarray (C,N,)
        a chord between points in the manifold
    proj_vec : np.ndarray (S,C,M)
        chords between corresponding points in the projected manifold

    Returns
    -------
    distortion : np.ndarray (S,)
        maximum distortion of chords
    """
    scale = np.sqrt(vec.size / proj_vec.shape[-1])
    return np.abs(scale * np.linalg.norm(proj_vec, axis=-1)
                  / np.linalg.norm(vec, axis=-1) - 1.).max(axis=-1)


def distortion_mfld(epsilon: np.ndarray,
                    mfld: np.ndarray,
                    proj_mflds: np.ndarray,
                    pinds: Pind,
                    chunk: int = 500):
    """
    Max distortion of all chords between points on the manifold

    Parameters
    ----------
    mfld[st...,i]
        phi_i(x[s],y[t],...),  (L,N),
        matrix of points on manifold as row vectors,
        i.e. flattened over intrinsic location indices
    proj_mfld[q,st...,i]
        phi_i(x[s],y[t],...),  (S,L,M),
        projected manifolds, first index is sample #
    pinds
        set of tuples of idxs for subregion (fL(fL-1)/2,)(2,)
    chunk
        chords are processed (vectorised) in chunks of this length.
        The different chunks are looped over.

    Modified
    --------
    epsilon
        max distortion of all chords (S,).
        Modified in place. Initial values are max distortion of tangent spaces.
    """
    for i in dbatch('(x1,x2)', 0, pinds.shape[0], chunk):
        pind = pinds[i].T  # (2,C)
        chord = mfld[pind[0]] - mfld[pind[1]]  # (C,N)
        proj_chord = proj_mflds[:, pind[0]] - proj_mflds[:, pind[1]]  # (S,C,M)
        np.maximum(epsilon, distortion_vec(chord, proj_chord),
                   out=epsilon)  # (S,)


def distortion_v(mfld: np.ndarray,
                 proj_mflds: np.ndarray,
                 proj_gmap: Sequence[np.ndarray],
                 region_inds: Sequence[Inds],
                 chunk: int = 500) -> np.ndarray:
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
        tuple of e_A^i(x[s],y[t],...),  (K,)(S,L,K,M),
        tuple members: gauss map of projected manifolds, 1sts index is sample #
    region_inds
        list of tuples of lists of arrays containing indices of: new points &
        new pairs in 1d and 2d subregions (#(V),2,#(K)),
        each element an array of indices (fL,) or (fL(fL-1)/2,)
    chunk
        chords are processed (vectorised) in chunks of this length.
        The different chunks are looped over.

    Returns
    -------
    epsilon = max distortion of all chords (#(K),#(V),S)
    """
    # tangent space distortions, (#(K),)(S,L)
    gdistn = ru.distortion_gmap(proj_gmap, mfld.shape[-1])

    distortion = np.empty((len(region_inds[0][0]),
                           len(region_inds),
                           proj_mflds.shape[0]))  # (#(K),#(V),S)

    for v, inds in denumerate('Vol', region_inds):
        for k, gdn, pts, pairs in denumerate('K', gdistn, inds[0], inds[1]):

            distortion[k, v] = gdn[:, pts].max(axis=-1)  # (S,)
            distortion_mfld(distortion[k, v], mfld, proj_mflds, pairs, chunk)

    # because each entry in region_inds  only contains new chords
    np.maximum.accumulate(distortion, axis=1, out=distortion)  # (#(K),#(V),S)

    return distortion


def distortion_m(mfld: np.ndarray,
                 gmap: np.ndarray,
                 proj_dims: np.ndarray,
                 uni_opts: Mapping[str, Real],
                 region_inds: Sequence[Inds]) -> np.ndarray:
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
    num_samp
        S, # samples of projectors for empirical distribution
    region_inds
        list of tuples of lists of arrays containing indices of: new points &
        new pairs in 1d and 2d subregions (#(V),2,#(K)),
        each element an array of indices (fL,) or (fL(fL-1)/2,)

    Returns
    -------
    epsilon = max distortion of chords for each (#(K),#(M),#(V),S)
    """
    # preallocate output. (#(K),#(M),#(V),S)
    distn = np.empty((len(region_inds[0][0]), len(proj_dims),
                      len(region_inds), uni_opts['samples']))

    batch = uni_opts['batch']
    for s in dbatch('Sample:', 0, uni_opts['samples'], batch):
        # projected manifold for each sampled proj, (S,Lx*Ly...,max(M))
        # gauss map of projected mfold for each proj, (#K,)(S,L,K,max(M))
        pmflds, pgmap = ru.project_mfld(mfld, gmap, proj_dims[-1], batch)

        # loop over M
        for i, M in denumerate('M', proj_dims):
            # distortions of all chords in (K-dim slice of) manifold
            distn[:, i, :, s] = distortion_v(mfld, pmflds[..., :M],
                                             [pgm[..., :M] for pgm in pgmap],
                                             region_inds, uni_opts['chunk'])
    return distn


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
