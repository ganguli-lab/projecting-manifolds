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
from typing import Sequence, Tuple, List, MutableSequence, Mapping, Dict
from numbers import Real
import itertools as it
from math import floor
from scipy.stats.mstats import gmean
import scipy.spatial.distance as scd
import numpy as np
from ..RandCurve import gauss_surf as gs
from ..disp_counter import denum, display_counter

Lind = np.ndarray  # Iterable[int]  # Set[int]
Pind = np.ndarray  # Iterable[Tuple[int, int]]  # Set[Tuple[int, int]]
Inds = Tuple[Sequence[Lind], Sequence[Pind]]

# K hard coded in: Options, make_surf,
# =============================================================================
# %%* generate manifold
# =============================================================================


def make_surf(ambient_dim: int,
              mfld_info: Mapping[str, Sequence[Real]],
              expand: int = 2) -> (np.ndarray, np.ndarray):
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
    """
    # Spatial frequencies used
    k_x, k_y = gs.spatial_freq(mfld_info['L'], mfld_info['num'], expand)
    # Fourier transform of embedding functions, (N,Lx,Ly/2)
    embed_ft = gs.random_embed_ft(ambient_dim, k_x, k_y, mfld_info['lambda'])
    # Fourier transform back to real space (N,Lx,Ly)
    emb = gs.embed(embed_ft)
    # find the image of the gauss map (push forward vielbein)
    grad = gs.embed_grad(embed_ft, k_x, k_y)
    # which elements to remove to select central region
    removex = (expand - 1) * mfld_info['num'][0] // 2
    removey = (expand - 1) * mfld_info['num'][1] // 2
    # throw out side regions, to lessen effects of periodicity
    mfld = emb[removex:-removex, removey:-removey, :]
    tang = grad[removex:-removex, removey:-removey, ...]
    return mfld, tang


def mfld_region(mfld: np.ndarray,
                gmap: np.ndarray,
                region_frac: float = 1.) -> (np.ndarray, np.ndarray):
    """
    Get central region of manifold

    Parameters
    ----------
    mfld[s,t,...,i]
        phi_i(x[s],y[t],...), (Lx,Ly,...,N)
        Embedding functions of random surface
    gmap/tang
        orthonormal/unnormalised basis for tangent space, (Lx,Ly,...,N,K)
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
            # deal with remove == 0 case
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


def mat_field_evals(mat_field: np.ndarray) -> np.ndarray:
    """eig
    """
    if mat_field.shape[-1] == 1:
        return mat_field.squeeze(-1)
    elif mat_field.shape[-1] == 2:
        return np.stack(gs.mat_field_evals(mat_field), axis=-1)
    return np.linalg.eigh(mat_field)[0]

# =============================================================================
# %%* region indexing
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
    [lin_ind1, lni_ind2]
        ndarray of indices of points on manifold, restricted to
        (1d slice of central region, 2d central region), (fLx, fL)
    [pinds1, pinds2]
        ndarray of indices of chords in condensed matrix from pdist,
        restricted to (1d slice of central region, 2d central region)
        (fLx(fLx-1)/2, fL(fL-1)/2)
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
    lin_pairs2 = np.array(list(it.combinations(lin_ind2, 2))).T
    lin_pairs1 = np.array(list(it.combinations(lin_ind1, 2))).T
    # indices in condensed matrix for chords in kept region
    pinds2 = (lin_pairs2[0] * (2 * n - lin_pairs2[0] - 3) // 2
              + lin_pairs2[1] - 1)
    pinds1 = (lin_pairs1[0] * (2 * n - lin_pairs1[0] - 3) // 2
              + lin_pairs1[1] - 1)
    return [lin_ind1, lin_ind2], [pinds1, pinds2]


def region_indices(shape: Sequence[int],
                   mfld_frac: float) -> (np.ndarray, np.ndarray):
    """
    Indices of points corresponding to the central region of the manifold.
    Smaller `mfld_frac` is guaranteed to return a subset of larger `mfld_frac`.

    Parameters
    ----------
    shape
        tuple of number of points along each dimension (K,)
    mfld_frac
        fraction of manifold to keep
    prev_1d, prev_2d
        set of indices already included

    Returns
    -------
    lin_ind1, lin_ind2
        set of indices of points on manifold, restricted to
        (1d slice of central region, 2d central region), (fLx, fL)
    """
    ranges = ()
    midranges = ()
    for siz in shape:
        # how many elements to remove?
        remove = floor((1. - mfld_frac)*siz)
#        """
        # how many of those to remove from start?
        removestart = remove // 2
        # which point to use for lower K?
        mid = siz // 2
        """
        # how many of those to remove from start?
        removestart = np.random.randint(remove + 1)
        # which point to use for lower K?
        mid = np.random.randint(siz)
        """
        # slice for region in eack dimension
        ranges += (np.arange(removestart, siz + removestart - remove),)
        # slice for point in each dimension, needed for lower K
        midranges += (np.array([mid]),)
    all_ranges = [ranges[:k] + midranges[k:] for k in range(1, len(shape) + 1)]
    # indices of region in after ravel
    lin_inds = tuple(np.ravel_multi_index(np.ix_(*range_k), shape).ravel()
                     for range_k in all_ranges)
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
            pind = []
            pind.extend(it.combinations(lind, 2))
            pind.extend(it.product(lind, sofar))
            # update set of old entries
            sofar = np.union1d(sofar, lind)
            # convert to arrays
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
    vec : np.ndarray (C,N,)
    proj_vec : np.ndarray (S,C,M)
    """
    scale = np.sqrt(vec.size / proj_vec.shape[-1])
    return np.abs(scale * np.linalg.norm(proj_vec, axis=-1)
                  / np.linalg.norm(vec, axis=-1) - 1.)


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
    for i in display_counter('(x1,x2)', 0, pinds.shape[0], chunk):
        pind = pinds[i:i+chunk].T  # (2,C)
        chord = mfld[pind[0]] - mfld[pind[1]]  # (C,N)
        proj_chord = proj_mflds[:, pind[0]] - proj_mflds[:, pind[1]]  # (S,C,M)
        np.maximum(epsilon,
                   distortion_vec(chord, proj_chord).max(axis=-1),  # (S,C)->S,
                   out=epsilon)  # (S,)


def distortion_v(mfld: np.ndarray,
                 proj_mflds: np.ndarray,
                 proj_gmap: Sequence[np.ndarray],
                 region_inds: MutableSequence[Inds],
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

    Returns
    -------
    epsilon = max distortion of all chords (#(K),#(V),S)
    """
    N = mfld.shape[-1]
    M = proj_mflds.shape[-1]

    # tangent space/projection angles, (#(K),)(S,L,K)
    cossq = [mat_field_evals(v @ v.swapaxes(-2, -1)) for v in proj_gmap]
    # tangent space distortions, (#(K),)(S,L)
    gdistn = [np.abs(np.sqrt(c * N / M) - 1).max(axis=-1) for c in cossq]

    distortion = np.empty((len(region_inds[0][0]),
                           len(region_inds),
                           proj_mflds.shape[0]))  # (#(K),#(V),S)

    for v, inds in denum('Vol', region_inds):
        for k, gdn, pts, pairs in denum('K', gdistn, inds[0], inds[1]):

            distortion[k, v] = gdn[:, pts].max(axis=-1)  # (S,)
            distortion_mfld(distortion[k, v], mfld, proj_mflds, pairs, chunk)

    # because each entry in region_inds  only contains new chords
    np.maximum.accumulate(distortion, axis=1, out=distortion)  # (#(K),#(V),S)

    return distortion


def distortion_m(mfld: np.ndarray,
                 gmap: np.ndarray,
                 proj_dims: np.ndarray,
                 uni_opts: Mapping[str, Real],
                 region_inds: MutableSequence[Inds]) -> np.ndarray:
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
    epsilon1d2d = max distortion of chords for each (#(K),#(M),#(V),S)
    """
    # preallocate output. (#(K),#(M),#(V),S)
    distn = np.empty((len(region_inds[0][0]), len(proj_dims),
                      len(region_inds), uni_opts['samples']))

    print(' Projecting', end='', flush=True)
    # sample projectors, (S,N,max(M))
    projs = make_basis(uni_opts['samples'], mfld.shape[-1], proj_dims[-1])
    # projected manifold for each sampled projector, (S,Lx*Ly...,max(M))
    proj_mflds = mfld @ projs
    # gauss map of projected manifold for each projector, (S,L,K,max(M))
    proj_gmap = [gmap[:, :k+1] @ projs[:, None] for k in range(gmap.shape[1])]
    print('\b \b' * len(' Projecting'), end='', flush=True)

    # loop over M
    for i, M in denum('M', proj_dims):
        # distortions of all chords in (K-dim slice of) manifold
        distn[:, i] = distortion_v(mfld, proj_mflds[..., :M],
                                   [pgmap[..., :M] for pgmap in proj_gmap],
                                   region_inds, uni_opts['chunk'])
    return distn


def distortion_V(ambient_dim: int,
                 proj_mflds: np.ndarray,
                 proj_gmap: Sequence[np.ndarray],
                 chordlen: np.ndarray,
                 region_inds: Sequence[Sequence[np.ndarray]]) -> (np.ndarray,
                                                                  np.ndarray):
    """


    Returns
    -------
    epsilon1d, epsilon2d = max distortion of all chords (#(V),S)
    """
    proj_dim = proj_mflds.shape[-1]

    # tangent space/projection angles, (#(K),)(S,L,K)
    cossq = [mat_field_evals(v @ v.swapaxes(-2, -1)) for v in proj_gmap]
    # tangent space distortions, (#(K),)(S,L)
    gdistn = [np.abs(np.sqrt(c * ambient_dim / proj_dim) - 1).max(axis=-1) for
              c in cossq]

    distortion = np.empty((len(region_inds[0][0]),
                           len(region_inds),
                           proj_mflds.shape[0]))  # (#(K),#(V),S)

    # loop over projected manifold for each sampled projection
    for i, pmfld in denum('Trial', proj_mflds):
        # length of chords in projected manifold
        projchordlen = scd.pdist(pmfld)
        # distortion of chords in 2d manifold
        distn_all = np.abs(np.sqrt(ambient_dim / proj_dim) *
                           projchordlen / chordlen - 1.)
        # maximum over kept region
        for j, inds, pinds in denum('Vol', region_inds[0], region_inds[1]):
            for k, ind, pind, gdn in denum('K', inds, pinds, gdistn):
                distortion[k, j, i] = np.maximum(distn_all[pind].max(axis=-1),
                                                 gdn[i, ind].max(axis=-1))
    return distortion


def distortion_M(mfld: np.ndarray,
                 gmap: np.ndarray,
                 proj_dims: np.ndarray,
                 uni_opts: Mapping[str, Real],
                 region_inds: MutableSequence[Inds]) -> np.ndarray:
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
    epsilon1d2d = max distortion of chords for each (#(K),#(M),#(V),S)
    """
    print('pdist', end='', flush=True)
    distn = np.empty((len(region_inds[0][0]), len(proj_dims),
                      len(region_inds), uni_opts['samples']))
    chordlen = scd.pdist(mfld)
    print('\b \b' * len('pdist'), end='', flush=True)

    print(' Projecting', end='', flush=True)
    # sample projectors, (S,N,max(M))
    projs = make_basis(uni_opts['samples'], mfld.shape[-1], proj_dims[-1])
    # projected manifold for each sampled projector, (S,Lx*Ly...,max(M))
    proj_mflds = mfld @ projs
    # gauss map of projected manifold for each projector, (S,L,K,max(M))
    proj_gmap = [gmap[:, :k+1] @ projs[:, None] for k in range(gmap.shape[1])]
    print('\b \b' * len(' Projecting'), end='', flush=True)

    # loop over M
    for i, M in denum('M', proj_dims):
        # distortions of all chords in (1d slice of, full 2d) manifold
        distn[:, i] = distortion_V(mfld.shape[-1], proj_mflds[..., :M],
                                   [pgmap[..., :M] for pgmap in proj_gmap],
                                   chordlen, region_inds)
    return distn


def distortion_percentile(distortions: np.ndarray,
                          prob: float) -> np.ndarray:
    """
    Calculate value of epsilon s.t. P(distortion > epsilon) = prob

    Parameters
    ----------
    distortions
        max distortion values for each sampled projection,
        ndarray (#(K),#(M),#(V),S)
    prob
        allowed failure probability

    Returns
    -------
    eps
        (1-prob)'th percentile of distortion, ndarray (#(K),#(M),#(V))
    """
    num_samp = distortions.shape[-1]
    cdf = np.linspace(0.5 / num_samp, 1. - 0.5/num_samp, num_samp)

    def func(distn):
        """interpolate cdf to find percentile"""
        return np.interp(1. - prob, cdf, distn)

    return np.apply_along_axis(func, -1, distortions)


def calc_reqd_m(epsilon: np.ndarray,
                proj_dims: np.ndarray,
                distortions: Sequence[np.ndarray]) -> np.ndarray:
    """
    Dimensionality of projection required to achieve distortion epsilon with
    probability (1-prob)

    Parameters
    ----------
    epsilon
        ndarray of allowed distortion(s), (#(epsilon),)
    proj_dims
        ndarray of M's, dimensionalities of projected space (#(M),)
    distortions
        (1-prob)'th percentile of distortion, ndarray  (#(K),#(M),#(V))

    Returns
    -------
    M
        required projection dimensionality, ndarray (#(K),#(epsilon),#(V))
    """

    # make sure it is strictly decreasing wrt M
    decr_eps = np.minimum.accumulate(distortions, axis=1)
    deps = np.cumsum((np.diff(decr_eps, axis=1) >= 0.) * 1.0e-6, axis=1)
    decr_eps -= np.pad(deps, ((0, 0), (1, 0), (0, 0)), 'constant')

#    def func(x): return np.interp(-np.asarray(epsilon), -x, proj_dims)
#    # linearly interpolate over epsilon to find M (need - so it increases)

    def func(distn):
        """linearly interpolate over vector 1/epsilon**2 to find M"""
        return np.interp(np.asarray(epsilon)**-2, distn**-2, proj_dims)

    # apply func to M axis
    return np.apply_along_axis(func, 1, decr_eps)


def reqd_proj_dim(mfld: np.ndarray,
                  gmap: np.ndarray,
                  epsilon: np.ndarray,
                  proj_dims: np.ndarray,
                  uni_opts: Mapping[str, Real],
                  region_inds: MutableSequence[Inds]) -> (np.ndarray,
                                                          np.ndarray):
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
        e_(A=0)^i must be parallel to d(phi^i)/dx^(a=0)
    epsilon
        allowed distortion, (#(epsilon),)
    prob
        allowed failure probability
    proj_dims
        ndarray of M's, dimensionalities of projected space (#(M),)
    num_samp
        number of samples of distortion for empirical distribution
    region_inds
        list of tuples of lists of arrays containing indices of: new points &
        new pairs in 1d and 2d subregions (#(V),2,#(K)),
        each element an array of indices (fL,) or (fL(fL-1)/2,)

    Returns
    -------
    M
        required projection dimensionality, ndarray (#(K),#(epsilon),#(V))
    distns
        (1-prob)'th percentile of distortion, for different M,
        ndarray (#(K),#(M),#(V))
    """
    mfld2, gmap2 = mfld_region(mfld, gmap)
    # sample projs and calculate distortion of all chords (#(M),S,L(L-1)/2)
    distortions = distortion_m(mfld2, gmap2, proj_dims, uni_opts, region_inds)
    # find max on each manifold, then find 1 - prob'th percentile, for each K,M
    eps = distortion_percentile(distortions, uni_opts['prob'])
    # find minimum M needed for epsilon, prob, for each K, epsilon
    reqd_m = calc_reqd_m(epsilon, proj_dims, eps)

    return reqd_m, eps

# =============================================================================
# %%* numeric data
# =============================================================================


def get_num_cmb(param_ranges: Mapping[str, np.ndarray],
                uni_opts: Mapping[str, Real],
                mfld_info: Mapping[str, Sequence[Real]]) -> (np.ndarray,
                                                             np.ndarray,
                                                             np.ndarray):
    """
    Calculate numerics as a function of N and V

    Parameters
    ----------
    epsilons
        ndarray of allowed distortions, (#(epsilon),)
    proj_dims
        ndarray of M's, dimensionalities of projected space (#(M),)
    ambient_dims
        ndarray of N's, dimensionality of ambient space (#(N),)
    mfld_fracs
        ndarray of fractions of ranges of intrinsic coords to keep (#V,)
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
    chunk
        chords are processed (vectorised) in chunks of this length.
        The different chunks are looped over.

    Returns
    -------
    proj_dim_num
        M for different N (#(K),#(epsilon),#(V),#(N))
    vols
        V^1/K, for each K, ndarray (#(K),#(V))
    distn_num
        (1-prob)'th percentile of distortion, for different N,
        ndarray (#(K),#(M),#(V),#(N))
    """

    proj_req = np.empty((len(mfld_info['L']), len(param_ranges['eps']),
                         len(param_ranges['Vfrac']), len(param_ranges['N'])))
    distn = np.empty((len(mfld_info['L']), len(param_ranges['M']),
                      len(param_ranges['Vfrac']), len(param_ranges['N'])))

    max_vols = [gmean(mfld_info['L'][:k]) / gmean(mfld_info['lambda'][:k])
                for k in range(1, 1 + len(mfld_info['lambda']))]
    vols = 2 * np.array(max_vols)[..., None] * param_ranges['Vfrac']

    # generate manifold
    print('mfld', end='', flush=True)
    mfld, tang = make_surf(param_ranges['N'][-1], mfld_info)
    print('\b \b' * len('mfld'), end='', flush=True)

    # indices for regions we keep
    region_inds = region_inds_list(mfld_info['num'], param_ranges['Vfrac'])

    Ms = param_ranges['M']

    for i, N in denum('N', param_ranges['N']):
        gmap = gs.vielbein(tang[..., :N, :])
        proj_req[..., i], distn[..., i] = reqd_proj_dim(mfld[..., :N], gmap,
                                                        param_ranges['eps'],
                                                        Ms[Ms <= N],
                                                        uni_opts, region_inds)

    return proj_req, vols, distn


def get_num_sep(param_ranges: Mapping[str, np.ndarray],
                uni_opts: Mapping[str, Real],
                mfld_info: Mapping[str, Sequence[Real]]) -> (np.ndarray,
                                                             np.ndarray,
                                                             np.ndarray,
                                                             np.ndarray):
    """
    Calculate all numerics as a function of N and V separately

    Parameters
    ----------
    epsilons
        ndarray of allowed distortions, (#(epsilon),)
    proj_dims
        ndarrays of M's, dimensionalities of projected space, (#(M),)
        tuple for (varying N, varying V)
    ambient_dims
        ndarrays of N's, dimensionality of ambient space,
        tuple for (varying N, varying V), first ndarray (#N,) second scalar
    mfld_fracs
        ndarray of fractions of ranges of intrinsic coords to keep (#V,)
    prob
        allowed failure probability
    num_samp
        number of samples of distortion for empirical distribution
    intrinsic_num
        tuple of numbers of sampling points on surface, (max(K),)
        tuple for (varying N, varying V)
    intr_range
        tuple of ranges of intrinsic coords, (max(K),):
            [-intr_range, intr_range]
        tuple for (varying N, varying V)
    width
        tuple of std devs of gauss cov along each intrinsic axis, (max(K),)
    chunk
        chords are processed (vectorised) in chunks of this length.
        The different chunks are looped over.

    Returns
    -------
    proj_dim_num, proj_dim_vol
        M for different N,V: ndarray (#(K),#(epsilon),#(N/V))
    vols_N, vols_V
         V^1/K for varying N & V, for each K, ndarray (#(K),) and (#(K), #(V))
    distn_N, distn_V
        (1-prob)'th percentile of distortion, for different N/V,
        ndarray (#(K),#(M),#(N/V))
    """

    proj_dim_num, vols_N, dist_N = get_num_cmb(endval(param_ranges, 'Vfrac'),
                                               uni_opts, mfld_info)
#    proj_dim_num = 1
#    vols_N = 1

    proj_dim_vol, vols_V, dist_V = get_num_cmb(endval(param_ranges, 'N'),
                                               uni_opts, mfld_info)

    return (proj_dim_num.squeeze(), proj_dim_vol.squeeze(), vols_N, vols_V,
            dist_N.squeeze(), dist_V.squeeze())


# =============================================================================
# %%* options
# =============================================================================


def default_options() -> (Dict[str, np.ndarray],
                          Dict[str, Real],
                          Dict[str, Tuple[Real, ...]]):
    """
    Default options for generating data

    Returns
    -------
    epsilons
        ndarray of allowed distortions (#epsilon)
    proj_dims
        ndarrays of M's, dimensionalities of projected space,
        tuple for (varying N, varying V), ndarrays: (#(M),)
    ambient_dims
        ndarrays of N's, dimensionality of ambient space,
        tuple for (varying N, varying V), first ndarray (#N,) second scalar
    mfld_fracs
        ndarray of fractions of ranges of intrinsic coords to keep (#V,)
    prob
        allowed failure probability
    num_samp
        number of samples of distortion for empirical distribution
    intrinsic_num
        tuple of numbers of sampling points on surface, (max(K),)
        tuple for (varying N, varying V)
    intrinsic_range
        tuple of ranges of intrinsic coords, (max(K),):
            [-intr_range, intr_range]
        tuple for (varying N, varying V)
    width
        tuple of std devs of gauss cov along each intrinsic axis, (max(K),)
    chunk
        chords are processed (vectorised) in chunks of this length.
        The different chunks are looped over.
    """
    # choose parameters
    np.random.seed(0)
    epsilons = np.array([0.2, 0.3, 0.4])
    proj_dims = np.linspace(5, 250, 50, dtype=int)
    # dimensionality of ambient space
    ambient_dims = np.logspace(8, 10, num=9, base=2, dtype=int)
    mfld_fracs = np.logspace(-1.5, 0, num=10, base=5)

    param_ranges = {'eps': epsilons,
                    'M': proj_dims,
                    'N': ambient_dims,
                    'Vfrac': mfld_fracs}

    uni_opts = {'prob': 0.05,
                'samples': 100,
                'chunk': 10000}
    mfld_info = {'num': (128, 128),  # number of points to sample
                 'L': (64.0, 64.0),  # x-coordinate lies between +/- this
                 'lambda': (8.0, 8.0)}  # correlation lengths

    return param_ranges, uni_opts, mfld_info


def quick_options() -> (Dict[str, np.ndarray],
                        Dict[str, Real],
                        Dict[str, Tuple[Real, ...]]):
    """
    Demo options for generating test data

    Returns
    -------
    epsilons
        ndarray of allowed distortions, (#(epsilon),)
    proj_dims
        ndarrays of M's, dimensionalities of projected space, (#(M),)
        tuple for (varying N, varying V)
    ambient_dims
        ndarrays of N's, dimensionality of ambient space,
        tuple for (varying N, varying V), first ndarray (#N,) second scalar
    mfld_fracs
        ndarray of fractions of ranges of intrinsic coords to keep (#V,)
    prob
        allowed failure probability
    num_samp
        number of samples of distortion for empirical distribution
    intrinsic_num
        tuple of numbers of sampling points on surface, (max(K),)
        tuple for (varying N, varying V)
    intrinsic_range
        tuple of ranges of intrinsic coords, (max(K),):
            [-intr_range, intr_range]
        tuple for (varying N, varying V)
    width
        tuple of std devs of gauss cov along each intrinsic axis, (max(K),)
    chunk
        chords are processed (vectorised) in chunks of this length.
        The different chunks are looped over.
    """
    # choose parameters
    np.random.seed(0)
    epsilons = np.array([0.2, 0.3])
    proj_dims = np.linspace(4, 200, 5, dtype=int)
    # dimensionality of ambient space
    amb_dims = np.logspace(np.log10(200), np.log10(400), num=3, dtype=int)
    mfld_fracs = np.logspace(-3, 0, num=4, base=2)

    param_ranges = {'eps': epsilons,
                    'M': proj_dims,
                    'N': amb_dims,
                    'Vfrac': mfld_fracs}

    uni_opts = {'prob': 0.05,
                'samples': 20,
                'chunk': 10000}
    mfld_info = {'num': (16, 32),  # number of points to sample
                 'L': (6.0, 10.0),  # x-coordinate lies between +/- this
                 'lambda': (1.0, 1.8)}  # correlation lengths

    return param_ranges, uni_opts, mfld_info


def endval(param_dict: Dict[str, np.ndarray],
           param: str) -> Dict[str, np.ndarray]:
    """Replace elements of array in dictionary with it's last element.
    """
    new_param_dict = param_dict.copy()
    new_param_dict[param] = param_dict[param][-1:]
    return new_param_dict

# =============================================================================
# %%* running code
# =============================================================================


def make_and_save(filename: str,
                  param_ranges: Mapping[str, np.ndarray],
                  uni_opts: Mapping[str, Real],
                  mfld_info: Mapping[str, Sequence[Real]]):
    """
    Generate data and save in ``.npz`` file

    Parameters
    ----------
    filename
        name of ``.npz`` file, w/o extension, for data
    epsilons
        ndarray of allowed distortions (#epsilon)
    proj_dims
        ndarrays of M's, dimensionalities of projected space, (2,)(#(M),)
        tuple for (varying N, varying V)
    ambient_dims
        ndarray of N's, dimensionality of ambient space,
        tuple for (varying N, varying V), first ndarray (#N,) second scalar
    mfld_fracs
        ndarray of fractions of ranges of intrinsic coords to keep (#V,)
    prob
        allowed failure probability
    num_samp
        number of samples of distortion for empirical distribution
    intrinsic_num
        tuple of numbers of sampling points on surface, (max(K),)
        tuple for (varying N, varying V)
    intr_range
        tuple of ranges of intrinsic coords, (max(K),):
            [-intr_range, intr_range]
        tuple for (varying N, varying V)
    width
        tuple of std devs of gauss cov along each intrinsic axis, (max(K),)
    chunk
        chords are processed (vectorised) in chunks of this length.
        The different chunks are looped over.

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
        tuple for (varying N, varying V), second one a scalar
    vols_N
         tuple of V^1/K, for each K,
    vols_V
        tuple of V^1/K, for each K, each member is an ndarray (#V)
    epsilons
        ndarray of allowed distortions (#epsilon)
    proj_dims
        ndarray of M's, dimensionalities of projected space,
        tuple for (varying N, varying V)
    distn_N, distn_V
        (1-prob)'th percentile of distortion, for different N,
        ndarray (#(K),#(M),#(N))
    distn_V
        (1-prob)'th percentile of distortion, for different V,
        ndarray (#(K),#(M),#(V))
    """
    # separate scans for N and V
    (M_num_N, M_num_V,
     vols_N, vols_V,
     dist_N, dist_V) = get_num_sep(param_ranges, uni_opts, mfld_info)

    np.savez_compressed(filename + '.npz',
                        num_N=M_num_N, num_V=M_num_V,
                        dist_N=dist_N, dist_V=dist_V,
                        vols_N=vols_N, vols_V=vols_V,
                        ambient_dims=param_ranges['N'],
                        epsilons=param_ranges['eps'],
                        proj_dims=param_ranges['M'],
                        prob=uni_opts['prob'])
#    # alternative, double scan
#    (M_num, vols, dist) = get_num_cmb(epsilons, proj_dims, ambient_dims,
#                                      mfld_fracs, prob, num_samp,
#                                      intrinsic_num, intr_range, width)
#    np.savez_compressed(filename + '.npz', M_num=M_num, prob=prob,
#                        ambient_dims=ambient_dims, vols=vols,
#                        epsilons=epsilons, proj_dims=proj_dims,
#                        dist=dist)

# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
