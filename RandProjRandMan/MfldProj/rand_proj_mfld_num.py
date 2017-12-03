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
import numpy as np
# import scipy.special as scf
from itertools import combinations as cmbi
from math import floor
import scipy.spatial.distance as scd
from typing import Sequence, Tuple, List
from ..RandCurve import gauss_curve as gc
from ..RandCurve import gauss_surf as gs
from ..disp_counter import denum
# from ..disp_counter import display_counter as disp


# =============================================================================
# generate manifold
# =============================================================================


def make_curve(ambient_dim: int,
               intrinsic_num: int,
               intr_range: float,
               width: float=1.0,
               expand: int=2) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Make random curve

    Parameters
    ----------
    ambient_dim
        N, dimensionality of ambient space
    intrinsic_num
        S, number of sampling points on curve
    intr_range
        L/2, range of intrinsic coord: [-intr_range, intr_range]
    width
        lambda, std dev of gaussian covariance

    Returns
    -------
    mfld[t,i]
        phi_i(x[t]), (Lx,N) Embedding fcs of random curve
    tang
        tang[t,i] = phi'_i(x[t])
    gmap
        normalised tangent vectors,
        gmap[t,i,A] = e_A^i(x[t]).
    """
    # Spatial frequencies used
    kx = gc.spatial_freq(intr_range, intrinsic_num, expand)
    # Fourier transform of embedding functions, (N,Lx/2)
    embed_ft = gc.random_embed_ft(ambient_dim, kx, width)
    # Fourier transform back to real space (N,Lx)
    emb = gc.embed(embed_ft)
    # find the image of the gauss map (push forward vielbein)
    grad = gc.embed_grad(embed_ft, kx)
    vbein = gs.vielbein(grad)
    # which elements to remove to select central region
    remove = (expand - 1) * intrinsic_num // 2
    # throw out side regions, to lessen effects of periodicity
    mfld = emb[remove:-remove, :]
    tang = grad[remove:-remove, ...]
    gmap = vbein[remove:-remove, ...]
    return mfld, tang[..., None], gmap[..., None]


def make_surf(ambient_dim: int,
              intrinsic_num: Sequence[int],
              intr_range: Sequence[float],
              width: Sequence[float]=(1.0, 1.0),
              expand: int=2) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Make random surface

    Parameters
    ----------
    ambient_dim
        N, dimensionality of ambient space
    intrinsic_num
        (Sx, Sy), tuple of numbers of sampling points on surface
    intr_range
        (Lx/2, Ly/2) tuple of ranges of intrinsic coords:
            [-intr_range, intr_range]
    width
        (lm_x, lm_y), tuple of std devs of gauss cov along each intrinsic axis
    expand
        max(Lx)/Lx = max(Ly)/Ly, integer > 1
        1 / fraction of ranges of intrinsic coords to keep

    Returns
    -------
    mfld[s,t,i]
        phi^i(x[s],y[t]) (Lx,Ly,N) Embedding fcns of random surface
    tang
        tang[s,t,i,a] = phi_a^i(x[s], y[t])
    gmap
        orthonormal basis for tangent space,
        gmap[s,t,i,A] = e_A^i(x[s], y[t]).
    """
    # Spatial frequencies used
    kx, ky = gs.spatial_freq(intr_range, intrinsic_num, expand)
    # Fourier transform of embedding functions, (N,Lx,Ly/2)
    embed_ft = gs.random_embed_ft(ambient_dim, kx, ky, width)
    # Fourier transform back to real space (N,Lx,Ly)
    emb = gs.embed(embed_ft)
    # find the image of the gauss map (push forward vielbein)
    grad = gs.embed_grad(embed_ft, kx, ky)
    vbein = gs.vielbein(grad)
    # which elements to remove to select central region
    removex = (expand - 1) * intrinsic_num[0] // 2
    removey = (expand - 1) * intrinsic_num[1] // 2
    # throw out side regions, to lessen effects of periodicity
    mfld = emb[removex:-removex, removey:-removey, :]
    tang = grad[removex:-removex, removey:-removey, ...]
    gmap = vbein[removex:-removex, removey:-removey, ...]
    return mfld, tang, gmap


def mfld_region(mfld: np.ndarray,
                gmap: np.ndarray,
                region_frac: float=1.) -> (np.ndarray, np.ndarray):
    """
    Get central region of manifold

    Parameters
    ----------
    mfld[s,t,...,i]
        phi_i(x[s],y[t],...), (Lx,Ly,...,N)
        Embedding functions of random surface
    gmap/tang
        orthonormal/unnormalised basis for tangent space,
        gmap[s,t,i,A] = e_A^i(x[s], y[t]),
        tang[s,t,i,a] = phi_a^i(x[s], y[t]).
    region_frac
        Lx/max(Lx) = Ly/max(Ly),
        fraction of ranges of intrinsic coords to keep (default: 1.)

    Returns
    -------
    new_mfld[st...,i]
        phi^i(x[s],y[t],...),  (L,N), L = Lx*Ly...,
        i.e. flattened over intrinsic location indices.
    new_gmap/newtang[st...,A,i]
        e_A^i(x[s],y[t],...) or phi_a^i(x[s], y[t]),  (L,K,N), L = Lx*Ly...,
        i.e. flattened over intrinsic location indices.
    """
    ambient_dim = mfld.shape[-1]
    mfld_dim = gmap.shape[-1]
    if region_frac >= 1.:
        new_mfld = mfld.reshape((-1, ambient_dim))
        new_gmap = gmap.reshape((-1, ambient_dim, mfld_dim))
    else:
        # find K
        slc = ()
        for k in range(mfld_dim):
            # which elements to remove to select central region
            remove = np.floor((1. - region_frac) * mfld.shape[k] / 2.,
                              dtype=int)
            # deal with rmove == 0 case
            if remove > 0:
                slc += (slice(remove, -remove),)
            else:
                slc += (slice(None),)
            # throw out side regions, to leave central region
        slc += (slice(None),) * 2
        new_mfld = mfld[slc[:-1]].reshape((-1, ambient_dim))
        new_gmap = gmap[slc].reshape((-1, ambient_dim, mfld_dim))
    return new_mfld, new_gmap.swapaxes(-2, -1)


def make_basis(num_samp: int,
               ambient_dim: int,
               proj_dim: int) -> np.ndarray:  # basis for random space
    """
    Generate orthonormal basis for subspace

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
        bases of subspaces for each sample, (S,N,M)
    """
    U = np.random.randn(num_samp, ambient_dim, proj_dim)
#    return np.array([np.linalg.qr(x)[0] for x in U])
    A = np.empty((num_samp, ambient_dim, proj_dim))
    for i, u in enumerate(U):
        # orthogonalise with Gram-Schmidt
        A[i] = np.linalg.qr(u)[0]
    return A


# =============================================================================
# calculate intermediaries
# =============================================================================


def region_squareform_inds(shape: Sequence[int],
                           mfld_frac: float) -> (np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray):
    """
    indices of condensed matrix returned by pdist corresponding to the
    central region of the manifold, and points on mfld

    Parameters
    ----------
    shape
        tuple of number of points along each dimension (K,)
    mfld_frac
        fraction of manifold to keep

    Returns
    -------
    pinds1, pinds2
        ndarray of indices of chords in condensed matrix from pdist,
        restricted to (1d slice of central region, 2d central region)
        (fLx(fLx-1)/2, fL(fL-1)/2)
    lin_ind1, lni_ind2
        ndarray of indices of points on manifold, restricted to
        (1d slice of central region, 2d central region), (fLx, fL)
    """
    ranges = ()
    midranges = ()
    for siz in shape:
        # how many elements to remove?
        remove = floor((1. - mfld_frac) * siz)
        """
        # how many of those to remove from start?
        removestart = remove // 2
        # which point to use for lower K?
        mid = siz // 2
        """
        # how many of those to remove from start?
        removestart = np.random.randint(remove + 1)
        # which point to use for lower K?
        mid = np.random.randint(siz)
#        """
        # slice for region in eack dimension
        ranges += (np.arange(removestart, siz + removestart - remove),)
        # slice for point in each dimension, needed for lower K
        midranges += (np.array([mid]),)
    ranges_1 = ranges[:-1] + midranges[-1:]
    # indices of region in after ravel
    lin_ind2 = np.ravel_multi_index(np.ix_(*ranges), shape).ravel()
    lin_ind1 = np.ravel_multi_index(np.ix_(*ranges_1), shape).ravel()
    n = np.prod(shape)
    # pairs of linear indices, i.e. ends of chords
    lin_pairs2 = np.array(list(cmbi(lin_ind2, 2)))
    lin_pairs1 = np.array(list(cmbi(lin_ind1, 2)))
    # indices in condensed matrix for chords in kept region
    pinds2 = (lin_pairs2[:, 0] * (2 * n - lin_pairs2[:, 0] - 3) // 2
              + lin_pairs2[:, 1] - 1)
    pinds1 = (lin_pairs1[:, 0] * (2 * n - lin_pairs1[:, 0] - 3) // 2
              + lin_pairs1[:, 1] - 1)
    return pinds1, pinds2, lin_ind1, lin_ind2


def region_squareform_mask(shape: Sequence[int],
                           mfld_frac: float) -> (np.ndarray, np.ndarray):
    """
    logical mask for condensed matrix returned by pdist corresponding to the
    central region of the manifold

    Parameters
    ----------
    shape
        tuple of nujmber of points along each dimension
    mfld_frac
        fraction of manifold to keep

    Returns
    -------
    pinds1, pinds2
        indices of chords in condensed matrix from pdist,
        restricted to (1d slice of central region, 2d central region)
    """
    ranges = ()
    midranges = ()
    for siz in shape:
        # how many elements to remove?
        remove = floor((1. - mfld_frac) * siz)
        """
        # how many of those to remove from start?
        removestart = remove // 2
        # which point to use for lower K?
        mid = siz // 2
        """
        # how many of those to remove from start?
        removestart = np.random.randint(remove + 1)
        # which point to use for lower K?
        mid = np.random.randint(siz + 1)
#        """
        # slice for region in eack dimension
        ranges += (np.arange(removestart, siz + removestart - remove),)
        # slice for point in each dimension, needed for lower K
        midranges += (np.array([mid]),)
    # mask along each dimension of kept region, broadcast
    sub_mask2 = np.zeros(shape, dtype=np.bool)
    sub_mask2[np.ix_(ranges)] = True
    sub_mask2 = sub_mask2.ravel()
    sub_mask1 = np.zeros(shape, dtype=np.bool)
    sub_mask1[np.ix_(ranges[:-1] + midranges[-1:])] = True
    sub_mask1 = sub_mask1.ravel()
    # mask for square distance matrix
    n = np.prod(shape)
    sq_mask2 = np.zeros((n, n), dtype=np.bool)
    sq_mask2[np.ix_(sub_mask2, sub_mask2)] = True
    sq_mask1 = np.zeros((n, n), dtype=np.bool)
    sq_mask1[np.ix_(sub_mask1, sub_mask1)] = True
    # upper triangle
    pinds2 = sq_mask2[np.tri(n, k=-1, dtype=np.bool).T]
    pinds1 = sq_mask1[np.tri(n, k=-1, dtype=np.bool).T]
    # independent elements
    return pinds1, pinds2


def region_inds_list(shape: Sequence[int],
                     mfld_frs: Sequence[float]) -> List[Sequence[np.ndarray]]:
    """
    List of index sets for different sized regions, each index set being the
    indices of condensed matrix returned by pdist corresponding to the
    central region of the manifold

    Parameters
    ----------
    shape
        tuple of nujmber of points along each dimension (K,)
    mfld_fracs
        list of fractions of manifold to keep (#(V))

    Returns
    -------
    region_inds
        list of tuples of ndarrays containing indices of: chords in
        condensed matrix from pdist, points on mfld
        each ndarray member: for a different fraction of manifold kept (#(V))
        each tuple: (1d chords, 2d chords, 1d points, 2d points)
    """
    region_inds = []
    for f in mfld_frs:
        # indices for regions we keep
        region_inds.append(region_squareform_inds(shape, f))
    return region_inds


def distortion(ambient_dim: int,
               proj_mflds: np.ndarray,
               proj_gmap: np.ndarray,
               chordlen: np.ndarray,
               region_inds: Sequence[Sequence[np.ndarray]]) -> (np.ndarray,
                                                                np.ndarray):
    """
    Distortion of all chords between points on the manifold

    Parameters
    ----------
    ambient_dim
        N, dimensionality of ambient space
    proj_mfld[q,st...,i]
        phi_i(x[s],y[t],...),  (S,L,M),
        projected manifolds, first index is sample #
    proj_gmap[k][q,st...,A,i]
        tuple of e_A^i(x[s],y[t],...),  (K,)(S,L,K,M),
        gauss map of projected manifolds, first index is sample #
    chordlen
        length of chords of mfld (L*(L-1)/2)
    region_inds
        list of tuples of idxs for 1d and 2d subregions (#(V),#(K)*2),
        each element an ndarray of indices (fL(fL-1)/2)

    Returns
    -------
    epsilon1d, epsilon2d = max distortion of all chords (#(V),S)
    """
    proj_dim = proj_mflds.shape[-1]
    num_samp = proj_mflds.shape[0]
    gram1d = np.linalg.norm(proj_gmap[0], axis=-1)  # (S,L)
    gram2d = proj_gmap[1] @ proj_gmap[1].swapaxes(-2, -1)  # (S,L,K,K)
    cossq = np.stack(gs.mat_field_evals(gram2d), axis=-1)  # (S,L,K)
    # tangent space distortions, (S,L)
    gdistn1d = np.abs(np.sqrt(ambient_dim / proj_dim) * gram1d - 1)
    gdistn2d = np.abs(np.sqrt(cossq * ambient_dim / proj_dim) - 1).max(axis=-1)

    distortion1d = np.empty((len(region_inds), num_samp))
    distortion2d = np.empty((len(region_inds), num_samp))
    # loop over projected manifold for each sampled projection
    for i, pmfld in denum('Trial', proj_mflds):
        # length of chords in projected manifold
        projchordlen = scd.pdist(pmfld)
        # distortion of chords in 2d manifold
        distn_all = np.abs(np.sqrt(ambient_dim / proj_dim) *
                           projchordlen / chordlen - 1.)
        # maximum over kept region
        for j, ind in denum('Vol', region_inds):
            distortion1d[j, i] = np.maximum(distn_all[ind[0]].max(axis=-1),
                                            gdistn1d[i, ind[2]].max(axis=-1))
            distortion2d[j, i] = np.maximum(distn_all[ind[1]].max(axis=-1),
                                            gdistn2d[i, ind[3]].max(axis=-1))
    return distortion1d, distortion2d


def distortion_M(mfld: np.ndarray,
                 gmap: np.ndarray,
                 proj_dims: np.ndarray,
                 num_samp: int,
                 region_inds: Sequence[Sequence[np.ndarray]]) -> (np.ndarray,
                                                                  np.ndarray):
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
    proj_dims
        ndarray of M's, dimensionalities of projected space (#(M),)
    num_samp
        S, # samples of projectors for empirical distribution
    region_inds
        list of tuples of idxs for 1d and 2d subregions (#(V),#(K)*2)

    Returns
    -------
    epsilon1d, epsilon2d = max distortion of chords for each (#(V),#(M),S)
    """
    print('pdist', end='', flush=True)
    distortion1d = np.empty((len(region_inds), len(proj_dims), num_samp))
    distortion2d = np.empty((len(region_inds), len(proj_dims), num_samp))
    chordlen = scd.pdist(mfld)
    print('\b \b' * len('pdist'), end='', flush=True)

    print('Projecting', end='', flush=True)
    ambient_dim = mfld.shape[-1]
    # sample projectors, (S,N,max(M))
    projs = make_basis(num_samp, ambient_dim, proj_dims[-1])
    # projected manifold for each sampled projector, (S,Lx*Ly...,max(M))
    proj_mflds = mfld @ projs
    # gauss map of projected manifold for each sampled projector,
    # (S,Lx*Ly...,K,max(M)) for K>1, for K=1 (S,Lx*Ly...,max(M))
    proj_gmap2 = gmap[None, ...] @ projs[:, None, ...]
    proj_gmap1 = gmap[..., 0, :] @ projs
    print('\b \b' * len('Projecting'), end='', flush=True)

    # loop over M
    for i, M in denum('M', proj_dims):
        # distortions of all chords in (1d slice of, full 2d) manifold
        (distortion1d[:, i, :],
         distortion2d[:, i, :]) = distortion(ambient_dim,
                                             proj_mflds[..., :M],
                                             (proj_gmap1[..., :M],
                                              proj_gmap2[..., :M]),
                                             chordlen,
                                             region_inds)
#        disp_ctr(i, len(proj_dims), ' M: ')
#    disp_ctr(len(proj_dims), len(proj_dims), ' M: ')
    return distortion1d, distortion2d


def distortion_percentile(distortions: np.ndarray,
                          prob: float) -> Sequence[np.ndarray]:
    """
    Calculate value of epsilon s.t. P(distortion > epsilon) = prob

    Parameters
    ----------
    distortions
        tuple of max distortion values for each K,
        each member an ndarray, (#(V),#(M),S)
    prob
        allowed failure probability

    Returns
    -------
    eps
        (1-prob)'th percentile of distortion, tuple for each K,
        each member, ndarray (#(K),)(#(V),#(M))
    """
    eps = ()
    # loop over K
    for distort in distortions:
        if distort.ndim == 2:
            distort.shape = (1,) + distort.shape
        # arrange samples in ascending oreder
        distort.sort(axis=-1)
        num_samp = distort.shape[-1]
        # values of cdf(epsilon) for each epsilon in distort
        cdf = np.linspace(0.5 / num_samp, 1. - 0.5/num_samp, num_samp)
        epsilons = np.empty(distort.shape[:-1])
        # loop over V
        for i, distV in enumerate(distort):
            # loop over M
            for j, distM in enumerate(distV):
                # interpolate to find epsilon s.t. cdf(epsilon) = 1 - prob
                epsilons[i, j] = np.interp(1. - prob, cdf, distM)
        eps += (epsilons,)
    return eps


def calc_reqd_M(epsilon: np.ndarray,
                proj_dims: np.ndarray,
                distortions: Sequence[np.ndarray]) -> np.ndarray:
    """
    Dimensionality of projection required to achieve distortion epsilon with
    probability (1-prob)

    Parameters
    ----------
    epsilon
        ndarray of allowed distortion(s) (#(e),)
    proj_dims
        ndarray of M's, dimensionalities of projected space (#(M),)
    distortions
        (1-prob)'th percentile of distortion,
        tuple for each K, each member is an ndarray  (#(V),#(M))

    Returns
    -------
    M
        required projection dimensionality, ndarray (#(K),#(V),#(epsilon))
    """

    reqd_Ms = ()
    # loop over K
    for distortion in distortions:
        if distortion.ndim == 1:
            distortion.shape = (1,) + distortion.shape
        reqd_M = np.empty((distortion.shape[0], len(epsilon)))
        # loop over V
        for i, epsilons in enumerate(distortion):
            # make sure it is strictly decreasing
            deps = np.diff(np.minimum.accumulate(epsilons))
            deps[deps >= 0] = -1.0e-6
            decr_eps = np.cumsum(np.concatenate(([epsilons[0]], deps)))
            # interpolate over M to find minimum M that gives desired epsilon
            reqd_M[i] = np.interp(-epsilon, -decr_eps, proj_dims)
        reqd_Ms += (reqd_M,)

    return np.array(reqd_Ms)


def reqd_proj_dim(mfld: np.ndarray,
                  gmap: np.ndarray,
                  epsilon: np.ndarray,
                  prob: float,
                  proj_dims: np.ndarray,
                  num_samp: int,
                  region_inds: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """
    Dimensionality of projection required to achieve distortion epsilon with
    probability (1-prob)

    Parameters
    ----------
    mfld[s,t,...,i]
        phi_i(x[s],y[t],...), (Lx,Ly,...,N)
        Embedding functions of random surface
    gmap
        orthonormal basis for tangent space, (Lx,Ly,N,K)
        gmap[s,t,i,A] = e_A^i(x[s], y[t]).
    epsilon
        allowed distortion (#(e),)
    prob
        allowed failure probability
    proj_dims
        ndarray of M's, dimensionalities of projected space (#(M),)
    num_samp
        number of samples of distortion for empirical distribution
    region_inds
        list of tuples of idxs for 1d and 2d subregions (#(V),#(K)*2)

    Returns
    -------
    M
        required projection dimensionality, ndarray (#(K),#(V),#(epsilon))
    """
#    gmap = gs.vielbein(tang)
    mfld2, gmap2 = mfld_region(mfld, gmap)
    # sample projs and calculate distortion of all chords (#(M),S,L(L-1)/2)
    distortions = distortion_M(mfld2, gmap2, proj_dims, num_samp, region_inds)
    # find max on each manifold, then find 1 - prob'th percentile, for each K,M
    eps = distortion_percentile(distortions, prob)
    # find minimum M needed for epsilon, prob, for each K, epsilon
    reqd_M = calc_reqd_M(epsilon, proj_dims, eps)

    return reqd_M
    """
    tang
        unnormalised basis for tangent space,
        tang[s,t,i,a] = phi_a^i(x[s], y[t]).
    """

# =============================================================================
# numeric data
# =============================================================================


def get_num_numeric(epsilons: np.ndarray,
                    proj_dims: np.ndarray,
                    ambient_dims: np.ndarray,
                    prob: float,
                    num_samp: int,
                    intrinsic_num: Sequence[int],
                    intr_range: Sequence[float],
                    width: Sequence[float]=(1.0, 1.0)) -> (np.ndarray,
                                                           np.ndarray):
    """
    Calculate numerics as a function of N

    Parameters
    ----------
    epsilons
        ndarray of allowed distortions (#(e),)
    proj_dims
        ndarray of M's, dimensionalities of projected space (#(e),)
    ambient_dims
        ndarray of N's, dimensionality of ambient space (#(e),)
    prob
        allowed failure probability
    num_samp
        number of samples of distortion for empirical distribution
    intrinsic_num
        tuple of numbers of sampling points on surface (K,)
    intr_range
        tuple of ranges of intrinsic coords: [-intr_range, intr_range]
    width
        tuple of std devs of gauss cov along each intrinsic axis, (K,)

    Returns
    -------
    proj_dim_num
        M for different N (#(K),#(epsilon),#(N))
    vols
        tuple of V^1/K, for each K, (#(K),)(1,)
    """
    vols1 = intr_range[0] * 4. / width[0]
    vols2 = np.sqrt(np.prod(intr_range) / np.prod(width)) * 4.

    proj_dim_num = np.empty((2, len(epsilons), len(ambient_dims)))

    print('mfld', end='', flush=True)
    mfld, tang = make_surf(ambient_dims[-1], intrinsic_num, intr_range,
                           width)[:2]
    print('\b \b' * len('mfld'), end='', flush=True)

#    region_inds = [region_squareform_inds(intrinsic_num, 1.)]
    # indices for regions we keep
    region_inds = region_inds_list(intrinsic_num, [1.])
    for i, N in enumerate(ambient_dims):
        print('N =', N)
        gmap = gs.vielbein(tang[..., :N, :])
        proj_dim_num[:, :, i] = reqd_proj_dim(mfld[..., :N], gmap,
                                              epsilons, prob, proj_dims,
                                              num_samp, region_inds)

    return proj_dim_num, np.array([vols1, vols2])


def get_vol_numeric(epsilons: np.ndarray,
                    proj_dims: np.ndarray,
                    ambient_dim: int,
                    mfld_fracs: np.ndarray,
                    prob: float,
                    num_samp: int,
                    intrinsic_num: Sequence[int],
                    intr_range: Sequence[float],
                    width: Sequence[float]=(1., 1.)) -> (np.ndarray,
                                                         np.ndarray):
    """
    Calculate numerics as a function of V

    Parameters
    ----------
    epsilons
        ndarray of allowed distortions
    proj_dims
        ndarray of M's, dimensionalities of projected space
    ambient_dim
        N, dimensionality of ambient space
    mfld_fracs
        ndarray of fractions of ranges of intrinsic coords to keep,
    prob
        allowed failure probability
    num_samp
        number of samples of distortion for empirical distribution
    intrinsic_num
        tuple of numbers of sampling points on surface
    intr_range
        tuple of ranges of intrinsic coords: [-intr_range, intr_range]
    width
        tuple of std devs of gauss cov along each intrinsic axis

    Returns
    -------
    proj_dim_vol
        M for different V, (#(K),#(epsilon),#(V))
    vols
        tuple of V^1/K, for each K, each member is an ndarray (#(K),)(#(V),)
    """

    proj_dim_vol = np.empty((2, len(epsilons), len(mfld_fracs)))

    vols1 = np.array(mfld_fracs) * 2. * intr_range[0] / width[0]
    vols2 = np.array(mfld_fracs) * 2. * np.sqrt(np.prod(intr_range) /
                                                np.prod(width))
    vols = np.stack((vols1, vols2))

    # generate manifold
    print('mfld', end='', flush=True)
    mfld, gmap = make_surf(ambient_dim, intrinsic_num, intr_range, width)[::2]
    print('\b \b' * len('mfld'), end='', flush=True)
#    # flatten over space and transpose
#    mfld2, gmap2 = mfld_region(mfld, gmap)

    # indices for regions we keep
    region_inds = region_inds_list(intrinsic_num, mfld_fracs)
#        # find minimum M needed for epsilon, prob, for each K, V, epsilon
    proj_dim_vol = np.array(reqd_proj_dim(mfld, gmap, epsilons, prob,
                                          proj_dims, num_samp, region_inds))
#    # sample projections and calculate distortion of all chords (#(V),#(M),S)
#    distn1, distn2 = distortion_M(mfld2, gmap2, proj_dims, num_samp,
#                                  region_inds)
#
#    for i, distn in enumerate(zip(distn1, distn2)):
#        # find max on each mfld, then find 1-prob'th percentile, for each K,M
#        eps = distortion_percentile(distn, prob)
#        # find minimum M needed for epsilon, prob, for each K, epsilon
#        proj_dim_vol[:, :, i] = calc_reqd_M(epsilons, proj_dims, eps)

    return proj_dim_vol.swapaxes(1, 2), vols


def get_all_num(epsilons: np.ndarray,
                proj_dims: Sequence[np.ndarray],
                ambient_dims: Tuple[np.ndarray, int],
                mfld_fracs: np.ndarray,
                prob: float,
                num_samp: int,
                intrinsic_num: Sequence[Sequence[int]],
                intr_range: Sequence[Sequence[float]],
                width: Sequence[float]=(1., 1.)) -> (np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray):
    """
    Calculate all numerics

    Parameters
    ----------
    epsilons
        ndarray of allowed distortions
    proj_dims
        ndarrays of M's, dimensionalities of projected space,
        tuple for varying N and V
    ambient_dims
        ndarrays of N's, dimensionality of ambient space,
        tuple for varying N and V, first one ndarray (#N,) second one a scalar
    mfld_fracs
        ndarray of fractions of ranges of intrinsic coords to keep (#V,)
    prob
        allowed failure probability
    num_samp
        number of samples of distortion for empirical distribution
    intrinsic_num
        tuple of numbers of sampling points on surface, (max(K),)
        tuple for varying N and V
    intr_range
        tuple of ranges of intrinsic coords, (max(K),):
            [-intr_range, intr_range]
        tuple for varying N and V
    width
        tuple of std devs of gauss cov along each intrinsic axis, (max(K),)

    Returns
    -------
    proj_dim_num, proj_dim_vol = M for different N,V and vols
    vols = tuple of tuples, for N & V, for each K, each: ndarray of V^1/K
    """

    proj_dim_num, vols_N = get_num_numeric(np.array(epsilons), proj_dims[0],
                                           ambient_dims[0], prob, num_samp,
                                           intrinsic_num[0],
                                           intr_range[0], width)
#    proj_dim_num = 1
#    vols_N = 1

    print('Varying volume...')
    proj_dim_vol, vols_V = get_vol_numeric(np.array(epsilons), proj_dims[1],
                                           ambient_dims[1], mfld_fracs, prob,
                                           num_samp, intrinsic_num[1],
                                           intr_range[1], width)

    return proj_dim_num, proj_dim_vol, vols_N, vols_V


# =============================================================================
# options
# =============================================================================


def default_options() -> (np.ndarray,
                          Sequence[np.ndarray],
                          Tuple[np.ndarray, int],
                          np.ndarray,
                          float,
                          int,
                          Sequence[Sequence[int]],
                          Sequence[Sequence[float]],
                          Sequence[float]):
    """
    Default options for generating data

    Returns
    -------
    epsilons
        ndarray of allowed distortions (#epsilon)
    proj_dims
        ndarrays of M's, dimensionalities of projected space,
        tuple for varying N and V
    ambient_dims
        ndarrays of N's, dimensionality of ambient space,
        tuple for varying N and V, first one ndarray (#N,) second one a scalar
    mfld_fracs
        ndarray of fractions of ranges of intrinsic coords to keep (#V,)
    prob
        allowed failure probability
    num_samp
        number of samples of distortion for empirical distribution
    intrinsic_num
        tuple of numbers of sampling points on surface, (max(K),)
        tuple for varying N and V
    intrinsic_range
        tuple of ranges of intrinsic coords, (max(K),):
            [-intr_range, intr_range]
        tuple for varying N and V
    width
        tuple of std devs of gauss cov along each intrinsic axis, (max(K),)
    """
    # choose parameters
    np.random.seed(0)
    epsilons = np.array([0.2, 0.3, 0.4])
    proj_dims = (np.linspace(5, 250, 50, dtype=int),
                 np.linspace(5, 250, 50, dtype=int))
    # dimensionality of ambient space
    ambient_dims = (np.logspace(8, 10, num=9, base=2, dtype=int), 1000)
    mfld_fracs = np.logspace(-1.5, 0, num=10, base=5)
    prob = 0.05
    num_samp = 100
    # number of points to sample
    intrinsic_num = ((128, 128), (128, 128))
    # x-coordinate lies between +/- this
    intrinsic_range = ((64.0, 64.0), (64.0, 64.0))
    width = (8.0, 8.0)

    return (epsilons, proj_dims, ambient_dims, mfld_fracs, prob, num_samp,
            intrinsic_num, intrinsic_range, width)


def quick_options() -> (np.ndarray,
                        Sequence[np.ndarray],
                        Tuple[np.ndarray, int],
                        np.ndarray,
                        float,
                        int,
                        Sequence[Sequence[int]],
                        Sequence[Sequence[float]],
                        Sequence[float]):
    """
    Demo options for generating test data

    Returns
    -------
    epsilons
        ndarray of allowed distortions (#epsilon)
    proj_dims
        ndarrays of M's, dimensionalities of projected space,
        tuple for varying N and V
    ambient_dims
        ndarrays of N's, dimensionality of ambient space,
        tuple for varying N and V, first one ndarray (#N,) second one a scalar
    mfld_fracs
        ndarray of fractions of ranges of intrinsic coords to keep (#V,)
    prob
        allowed failure probability
    num_samp
        number of samples of distortion for empirical distribution
    intrinsic_num
        tuple of numbers of sampling points on surface, (max(K),)
        tuple for varying N and V
    intrinsic_range
        tuple of ranges of intrinsic coords, (max(K),):
            [-intr_range, intr_range]
        tuple for varying N and V
    width
        tuple of std devs of gauss cov along each intrinsic axis, (max(K),)
    """
    # choose parameters
    np.random.seed(0)
    epsilons = np.array([0.2, 0.3])
    proj_dims = (np.linspace(4, 200, 5, dtype=int),
                 np.linspace(4, 200, 5, dtype=int))
    # dimensionality of ambient space
    ambient_dims = (np.logspace(np.log10(200), np.log10(400), num=3,
                                dtype=int), 200)
    mfld_fracs = np.logspace(-3, 0, num=4, base=2)
    prob = 0.05
    num_samp = 20
    # number of points to sample
    intrinsic_num = ((16, 32), (16, 32))
    # x-coordinate lies between +/- this
    intrinsic_range = ((6.0, 10.0), (6.0, 10.0))
    width = (1.0, 1.8)

    return (epsilons, proj_dims, ambient_dims, mfld_fracs, prob, num_samp,
            intrinsic_num, intrinsic_range, width)


# =============================================================================
# running code
# =============================================================================


def make_and_save(filename: str,
                  epsilons: np.ndarray,
                  proj_dims: Sequence[np.ndarray],
                  ambient_dims: Tuple[np.ndarray, int],
                  mfld_fracs: np.ndarray,
                  prob: float,
                  num_samp: int,
                  intrinsic_num: Sequence[Sequence[int]],
                  intr_range: [Sequence[float]],
                  width: Sequence[float]=(1.0, 1.0)):  # generate data and save
    """
    Generate data and save in ``.npz`` file

    Parameters
    ----------
    filename
        name of ``.npz`` file, w/o extension, for data
    epsilons
        ndarray of allowed distortions (#epsilon)
    proj_dims
        ndarray of M's, dimensionalities of projected space,
        tuple for varying N and V
    ambient_dims
        ndarray of N's, dimensionality of ambient space,
        tuple for varying N and V, first one ndarray (#N,) second one a scalar
    mfld_fracs
        ndarray of fractions of ranges of intrinsic coords to keep (#V,)
    prob
        allowed failure probability
    num_samp
        number of samples of distortion for empirical distribution
    intrinsic_num
        tuple of numbers of sampling points on surface, (max(K),)
        tuple for varying N and V
    intr_range
        tuple of ranges of intrinsic coords, (max(K),):
            [-intr_range, intr_range]
        tuple for varying N and V
    width
        tuple of std devs of gauss cov along each intrinsic axis, (max(K),)

    Returns
    -------
    None, but saves .npz file (everything converted to ndarray) with fields:

    num_N
        values of M when varying N, ndarray (#K,#epsilon,#N)
    num_V
        values of M when varying V, ndarray (#K,#epsilon,#N)
    prob
        allowed failure probability
    ambient_dims
        ndarray of N's, dimensionality of ambient space, ((#N,), 1)
        tuple for varying N and V, second one a scalar
    vols_N
         tuple of V^1/K, for each K,
    vols_V
        tuple of V^1/K, for each K, each member is an ndarray (#V)
    epsilons
        ndarray of allowed distortions (#epsilon)
    """
    M_num_N, M_num_V, vols_N, vols_V = get_all_num(epsilons, proj_dims,
                                                   ambient_dims, mfld_fracs,
                                                   prob, num_samp,
                                                   intrinsic_num, intr_range,
                                                   width)

    np.savez_compressed(filename + '.npz', num_N=M_num_N, num_V=M_num_V,
                        prob=prob, ambient_dims=ambient_dims, vols_N=vols_N,
                        vols_V=vols_V, epsilons=epsilons)
#                       LGG_N=M_LGG_N, LGG_V=M_LGG_V, BW_N=M_BW_N, BW_V=M_BW_V)


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
