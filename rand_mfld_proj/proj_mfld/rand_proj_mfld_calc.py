# -*- coding: utf-8 -*-
# =============================================================================
# Created on Fri Jan 12 18:34:04 2018
#
# @author: Subhy
#
# Module rand_proj_mfld_calc
# =============================================================================
"""
Calculation of distribution of maximum distortion of Gaussian random manifolds
under random projections

Functions
=========
region_inds_list
    indices of points on mfld and condensed matrix returned by pdist
    corresponding to the central region of the manifold
distortion_m
    Maximum distortion of all chords between points on the manifold,
    sampling projectors, for each V, M
"""
from typing import Sequence, List, Mapping, Tuple
from numbers import Real
import numpy as np
import scipy.spatial.distance as scd

from ..iter_tricks import dbatch, denumerate, dcontext, rdenumerate
from ..larray import larray
from . import rand_proj_mfld_util as ru
from . import rand_proj_mfld_mem as rm

Lind = np.ndarray  # Iterable[int]  # Set[int]
Pind = np.ndarray  # Iterable[Tuple[int, int]]  # Set[Tuple[int, int]]
Inds = Tuple[Lind, Pind]

# =============================================================================
# %%* region indexing
# =============================================================================


def region_squareform_inds(shape: Sequence[int],
                           mfld_frac: float) -> List[Inds]:
    """
    indices of points on mfld and condensed matrix returned by pdist
    corresponding to the central region of the manifold

    Parameters
    ----------
    shape
        tuple of number of points along each dimension (max(K),)
    mfld_frac
        fraction of manifold to keep

    Returns
    -------
    lin_inds
        ndarray of indices of points on manifold, restricted to
        K-d central region, (#(K),)((fLx)^K)
    pinds
        ndarray of indices of chords in condensed matrix from pdist,
        restricted to K-d central region, (#(K),)((fLx)^K ((fLx)^K - 1)/2,)
    """
    pnum = np.prod(shape)
    # indices of region in after ravel
    lin_inds = rm.region_indices(shape, mfld_frac, random=True)
    # pairs of linear indices, i.e. ends of chords
    lin_pairs = [ru.pairs(lind) for lind in lin_inds]
    # indices in condensed matrix for chords in kept region
    pinds = [lpr[0]*(2*pnum - lpr[0] - 3)//2 + lpr[1] - 1 for lpr in lin_pairs]
    return [inds for inds in zip(lin_inds, pinds)]


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
        list of lists of tuples of arrays containing indices of: points & pairs
        in K-d subregions (#(V),)(#(K),)(2,)
        each element: array of indices ((fLx)^K,) or ((fLx)^K ((fLx)^K - 1)/2,)
    """
    return [region_squareform_inds(shape, frac) for frac in mfld_fs]


# =============================================================================
# %%* distortion calculations
# =============================================================================


def distortion_v(ambient_dim: int,
                 pmf_bundle: Tuple[larray, Sequence[larray]],
                 chordlen: np.ndarray,
                 region_inds: Sequence[Sequence[Inds]]) -> np.ndarray:
    """
    Max distortion of all tangent vectors and chords between points in various
    regions manifold, for all V

    Parameters
    ----------
    ambient_dim
        dimensionality of ambient space
    pmf_bundle
        Tuple: (proj_mfld, proj_gmap)
    proj_mfld[q,st...,i]
        phi_i(x[s],y[t],...),  (S,L,M),
        projected manifolds, first index is sample #
    proj_gmap[k][q,st...,A,i]
        tuple of e_A^i(x[s],y[t],...),  (K,)(S,L,K,M),
        tuple members: gauss map of projected manifolds, 1sts index is sample #
    region_inds
        list of lists of tuples of arrays containing indices of: new points &
        new pairs in 1d and 2d subregions (#(V),2,#(K)),
        each element an array of indices (fL,) or (fL(fL-1)/2,)

    Returns
    -------
    epsilon
        max distortion of all chords (#(K),#(V),S)
    """
    proj_dim = pmf_bundle[0].shape[-1]

    # tangent space distortions, (#(K),)(S,L)
    gdistn = rm.distortion_gmap(pmf_bundle[1], ambient_dim)

    distortion = np.empty((len(region_inds[0][0]),
                           len(region_inds),
                           pmf_bundle[0].shape[0]))  # (#(K),#(V),S)

    # loop over projected manifold for each sampled projection
    for s, pmfld in denumerate('Trial', pmf_bundle[0]):
        with dcontext('pdist'):
            # length of chords in projected manifold
            projchordlen = scd.pdist(pmfld)
        # distortion of chords in 2d manifold
        distn_all = np.abs(np.sqrt(ambient_dim / proj_dim) *
                           projchordlen / chordlen - 1.)
        # maximum over kept region
        for v, inds in denumerate('Vol', region_inds):
            for k, (lnd, pnd), gdn in denumerate('K', inds, gdistn):
                distortion[k, v, s] = np.fmax(distn_all[pnd].max(axis=-1),
                                              gdn[s, lnd].max(axis=-1))
    return distortion


def distortion_m(mfld_bundle: Tuple[larray, larray],
                 proj_dims: np.ndarray,
                 uni_opts: Mapping[str, Real],
                 region_inds: Sequence[Sequence[Inds]]) -> np.ndarray:
    """
    Maximum distortion of all chords between points on the manifold,
    sampling projectors, for each V, M

    Parameters
    ----------
    mfld_bundle
        Tuple: (mfld, gmap)
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
    with dcontext('pdist'):
        chordlen = scd.pdist(mfld_bundle[0])

    batch = uni_opts['batch']
    for s in dbatch('Sample', 0, uni_opts['samples'], batch):
        # projected manifold for each sampled proj, (S,Lx*Ly...,max(M))
        # gauss map of projected mfold for each proj, (#K,)(S,L,K,max(M))
        pmflds, pgmaps = rm.project_mfld(mfld_bundle, proj_dims[-1], batch)

        # loop over M
        for i, M in rdenumerate('M', proj_dims):
            pmf_bundle = (pmflds[..., :M], [pgm[..., :M] for pgm in pgmaps])
            # distortions of all chords in (1d slice of, full 2d) manifold
            distn[:, i, :, s] = distortion_v(mfld_bundle[0].shape[-1],
                                             pmf_bundle, chordlen, region_inds)
    return distn
