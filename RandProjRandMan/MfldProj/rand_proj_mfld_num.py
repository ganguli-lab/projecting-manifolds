# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:29:22 2017

@author: Subhy based on Peiran's Matlab code

Compute distribution of maximum distortion of Gaussian random manifolds under
random projections

Functions
=========
get_num_cmb
    calculate all numeric quantities, varying N and V together
get_num_sep
    calculate all numeric quantities, varying N and V separately
default_options
    default options for long numerics for paper
quick_options
    default options for quick numerics for demo
make_and_save
    generate data and save npz file
"""
from typing import Sequence, Tuple, Mapping, Dict
from numbers import Real
from scipy.stats.mstats import gmean
import numpy as np

from ..RandCurve import gauss_surf as gs
from ..iter_tricks import dcontext, rdenumerate
from . import rand_proj_mfld_calc as rc
from . import rand_proj_mfld_util as ru


# K hard coded in: Options, make_surf, get_num_cmb calls: gs.vielbein
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
    mfld_info
            dict of parameters for manifold sampling, with fields:
        num
            tuple of numbers of sampling points on surface, (max(K),)
        L
            tuple of ranges of intrinsic coords, (max(K),):
                [-intr_range, intr_range]
            tuple for (varying N, varying V)
        lambda
            tuple of std devs of gauss cov along each intrinsic axis, (max(K),)
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
    mfld = emb[removex:-removex, removey:-removey]
    tang = grad[removex:-removex, removey:-removey]
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


# =============================================================================
# %%* distortion calculations
# =============================================================================


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
        ndarray of allowed distortion(s), (#(e),)
    proj_dims
        ndarray of M's, dimensionalities of projected space (#(M),)
    distortions
        (1-prob)'th percentile of distortion, ndarray  (#(K),#(M),#(V))

    Returns
    -------
    M
        required projection dimensionality, ndarray (#(K),#(e),#(V))
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
                  param_ranges: Mapping[str, np.ndarray],
                  uni_opts: Mapping[str, Real]) -> (np.ndarray, np.ndarray):
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
    param_ranges
            dict of parameter ranges, with fields:
        epsilons : np.ndarray (#(e),)
            ndarray of allowed distortions
        proj_dims : np.ndarray (#(M),)
            ndarray of M's, dimensionalities of projected space,
        ambient_dims : np.ndarray (#(N),)
            ndarray of N's, dimensionality of ambient space,
        mfld_fracs : np.ndarray (#(V),)
            ndarray of fractions of ranges of intrinsic coords to keep
    uni_opts
            dict of scalar options, used for all parameter values, with fields:
        prob
            allowed failure probability
        num_samp
            number of samples of distortion for empirical distribution
        batch
            sampled projections are processed in batches of this length.
            The different batches are looped over (mem version).
        chunk
            chords are processed (vectorised) in chunks of this length.
            The different chunks are looped over (mem version).

    Returns
    -------
    M
        required projection dimensionality, ndarray (#(K),#(e),#(V))
    distns
        (1-prob)'th percentile of distortion, for different M,
        ndarray (#(K),#(M),#(V))
    """
    Ms = param_ranges['M'][param_ranges['M'] <= mfld.shape[-1]]
    with dcontext('flatten'):
        # flatten location indices
        mfld2, gmap2 = mfld_region(mfld, gmap)
    with dcontext('inds'):
        # indices for regions we keep
        region_inds = rc.region_inds_list(mfld.shape[:-1], param_ranges['Vfr'])

    # sample projs and calculate distortion of all chords (#(M),S,L(L-1)/2)
    distortions = rc.distortion_m(mfld2, gmap2, Ms, uni_opts, region_inds)
    # find max on each manifold, then find 1 - prob'th percentile, for each K,M
    eps = distortion_percentile(distortions, uni_opts['prob'])
    # find minimum M needed for epsilon, prob, for each K, epsilon
    reqd_m = calc_reqd_m(param_ranges['eps'], Ms, eps)

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
    param_ranges
            dict of parameter ranges, with fields:
        epsilons : np.ndarray (#(e),)
            ndarray of allowed distortions
        proj_dims : np.ndarray (#(M),)
            ndarray of M's, dimensionalities of projected space,
        ambient_dims : np.ndarray (#(N),)
            ndarray of N's, dimensionality of ambient space,
        mfld_fracs : np.ndarray (#(V),)
            ndarray of fractions of ranges of intrinsic coords to keep
    uni_opts
            dict of scalar options, used for all parameter values, with fields:
        prob
            allowed failure probability
        num_samp
            number of samples of distortion for empirical distribution
        batch
            sampled projections are processed in batches of this length.
            The different batches are looped over (mem version).
        chunk
            chords are processed (vectorised) in chunks of this length.
            The different chunks are looped over (mem version).
    mfld_info
            dict of parameters for manifold sampling, with fields:
        num
            tuple of numbers of sampling points on surface, (max(K),)
        L
            tuple of ranges of intrinsic coords, (max(K),):
                [-intr_range, intr_range]
            tuple for (varying N, varying V)
        lambda
            tuple of std devs of gauss cov along each intrinsic axis, (max(K),)

    Returns
    -------
    proj_dim_num
        M for different N (#(K),#(epsilon),#(V),#(N))
    distn_num
        (1-prob)'th percentile of distortion, for different N,
        ndarray (#(K),#(M),#(V),#(N))
    vols
        V^1/K, for each K, ndarray (#(K),#(V))
    """

    proj_req = np.empty((len(mfld_info['L']), len(param_ranges['eps']),
                         len(param_ranges['Vfr']), len(param_ranges['N'])))
    distn = np.empty((len(mfld_info['L']), len(param_ranges['M']),
                      len(param_ranges['Vfr']), len(param_ranges['N'])))

    max_vols = [gmean(mfld_info['L'][:k]) / gmean(mfld_info['lambda'][:k])
                for k in range(1, 1 + len(mfld_info['lambda']))]
    vols = 2 * np.array(max_vols)[..., None] * param_ranges['Vfr']

    # generate manifold
    with dcontext('mfld'):
        mfld, tang = make_surf(param_ranges['N'][-1], mfld_info)

    for i, N in rdenumerate('N', param_ranges['N']):
        gmap = gs.vielbein(tang[..., :N, :])
        proj_req[..., i], distn[..., i] = reqd_proj_dim(mfld[..., :N], gmap,
                                                        param_ranges, uni_opts)

    return proj_req, distn, vols


def get_num_sep(param_ranges: Mapping[str, np.ndarray],
                uni_opts: Mapping[str, Real],
                mfld_info: Mapping[str, Sequence[Real]]) -> (np.ndarray,
                                                             np.ndarray,
                                                             np.ndarray,
                                                             np.ndarray,
                                                             np.ndarray):
    """
    Calculate all numerics as a function of N and V separately

    Parameters
    ----------
    param_ranges
            dict of parameter ranges, with fields:
        epsilons : np.ndarray (#(e),)
            ndarray of allowed distortions
        proj_dims : np.ndarray (#(M),)
            ndarray of M's, dimensionalities of projected space,
        ambient_dims : np.ndarray (#(N),)
            ndarray of N's, dimensionality of ambient space,
        mfld_fracs : np.ndarray (#(V),)
            ndarray of fractions of ranges of intrinsic coords to keep
    uni_opts
            dict of scalar options, used for all parameter values, with fields:
        prob
            allowed failure probability
        num_samp
            number of samples of distortion for empirical distribution
        batch
            sampled projections are processed in batches of this length.
            The different batches are looped over (mem version).
        chunk
            chords are processed (vectorised) in chunks of this length.
            The different chunks are looped over (mem version).
    mfld_info
            dict of parameters for manifold sampling, with fields:
        num
            tuple of numbers of sampling points on surface, (max(K),)
        L
            tuple of ranges of intrinsic coords, (max(K),):
                [-intr_range, intr_range]
            tuple for (varying N, varying V)
        lambda
            tuple of std devs of gauss cov along each intrinsic axis, (max(K),)

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

    proj_dim_num, dist_N = get_num_cmb(ru.endval(param_ranges, 'Vfr'),
                                       uni_opts, mfld_info)[:2]
#    proj_dim_num = 1
#    vols_N = 1

    proj_dim_vol, dist_V, vols = get_num_cmb(ru.endval(param_ranges, 'N'),
                                             uni_opts, mfld_info)

    return (proj_dim_num.squeeze(), proj_dim_vol.squeeze(), dist_N.squeeze(),
            dist_V.squeeze(), vols)


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
    param_ranges
            dict of parameter ranges, with fields:
        epsilons : np.ndarray (#(e),)
            ndarray of allowed distortions
        proj_dims : np.ndarray (#(M),)
            ndarray of M's, dimensionalities of projected space,
        ambient_dims : np.ndarray (#(N),)
            ndarray of N's, dimensionality of ambient space,
        mfld_fracs : np.ndarray (#(V),)
            ndarray of fractions of ranges of intrinsic coords to keep
    uni_opts
            dict of scalar options, used for all parameter values, with fields:
        prob
            allowed failure probability
        num_samp
            number of samples of distortion for empirical distribution
        batch
            sampled projections are processed in batches of this length.
            The different batches are looped over (mem version).
        chunk
            chords are processed (vectorised) in chunks of this length.
            The different chunks are looped over (mem version).
    mfld_info
            dict of parameters for manifold sampling, with fields:
        num
            tuple of numbers of sampling points on surface, (max(K),)
        L
            tuple of ranges of intrinsic coords, (max(K),):
                [-intr_range, intr_range]
            tuple for (varying N, varying V)
        lambda
            tuple of std devs of gauss cov along each intrinsic axis, (max(K),)
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
                    'Vfr': mfld_fracs}

    uni_opts = {'prob': 0.05,
                'samples': 100,
                'chunk': 10000,
                'batch': 25}
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
    param_ranges
            dict of parameter ranges, with fields:
        epsilons : np.ndarray (#(e),)
            ndarray of allowed distortions
        proj_dims : np.ndarray (#(M),)
            ndarray of M's, dimensionalities of projected space,
        ambient_dims : np.ndarray (#(N),)
            ndarray of N's, dimensionality of ambient space,
        mfld_fracs : np.ndarray (#(V),)
            ndarray of fractions of ranges of intrinsic coords to keep
    uni_opts
            dict of scalar options, used for all parameter values, with fields:
        prob
            allowed failure probability
        num_samp
            number of samples of distortion for empirical distribution
        batch
            sampled projections are processed in batches of this length.
            The different batches are looped over (mem version).
        chunk
            chords are processed (vectorised) in chunks of this length.
            The different chunks are looped over (mem version).
    mfld_info
            dict of parameters for manifold sampling, with fields:
        num
            tuple of numbers of sampling points on surface, (max(K),)
        L
            tuple of ranges of intrinsic coords, (max(K),):
                [-intr_range, intr_range]
            tuple for (varying N, varying V)
        lambda
            tuple of std devs of gauss cov along each intrinsic axis, (max(K),)
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
                    'Vfr': mfld_fracs}

    uni_opts = {'prob': 0.05,
                'samples': 20,
                'chunk': 10000,
                'batch': 10}
    mfld_info = {'num': (64, 64),  # number of points to sample
                 'L': (64.0, 64.0),  # x-coordinate lies between +/- this
                 'lambda': (8.0, 8.0)}  # correlation lengths

    return param_ranges, uni_opts, mfld_info

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
    param_ranges
            dict of parameter ranges, with fields:
        epsilons : np.ndarray (#(e),)
            ndarray of allowed distortions
        proj_dims : np.ndarray (#(M),)
            ndarray of M's, dimensionalities of projected space,
        ambient_dims : np.ndarray (#(N),)
            ndarray of N's, dimensionality of ambient space,
        mfld_fracs : np.ndarray (#(V),)
            ndarray of fractions of ranges of intrinsic coords to keep
    uni_opts
            dict of scalar options, used for all parameter values, with fields:
        prob
            allowed failure probability
        num_samp
            number of samples of distortion for empirical distribution
        batch
            sampled projections are processed in batches of this length.
            The different batches are looped over (mem version).
        chunk
            chords are processed (vectorised) in chunks of this length.
            The different chunks are looped over (mem version).
    mfld_info
            dict of parameters for manifold sampling, with fields:
        num
            tuple of numbers of sampling points on surface, (max(K),)
        L
            tuple of ranges of intrinsic coords, (max(K),):
                [-intr_range, intr_range]
            tuple for (varying N, varying V)
        lambda
            tuple of std devs of gauss cov along each intrinsic axis, (max(K),)

    Returns
    -------
    None, but saves .npz file (everything converted to ndarray) with fields:

    num
        values of M when varying K,epsilon,V,N, ndarray (#K,#(e),#(V),#(N))
    prob
        allowed failure probability
    ambient_dims
        ndarray of N's, dimensionality of ambient space, (#N,)
    vols
        V^1/K, for each K, each member is an ndarray (#(K),#(V))
    epsilons
        ndarray of allowed distortions (#(e),)
    proj_dims
        ndarray of M's, dimensionalities of projected space,
    distn
        (1-prob)'th percentile of distortion, for different N, V, M, K
        ndarray (#(K),#(M),#(V),#(N))
    """
    if uni_opts['samples'] % uni_opts['batch'] != 0:
        msg = 'samples must be divisible by batches. samples: {}, batch: {}.'
        raise ValueError(msg.format(uni_opts['samples'], uni_opts['batch']))
#    """
#    Returns
#    -------
#    None, but saves .npz file (everything converted to ndarray) with fields:
#
#    num_N
#        values of M when varying N, ndarray (#K,#epsilon,#N)
#    num_V
#        values of M when varying V, ndarray (#K,#epsilon,#N)
#    prob
#        allowed failure probability
#    ambient_dims
#        ndarray of N's, dimensionality of ambient space, ((#N,), 1)
#        tuple for (varying N, varying V), second one a scalar
#    vols_N
#         tuple of V^1/K, for each K,
#    vols_V
#        tuple of V^1/K, for each K, each member is an ndarray (#V)
#    epsilons
#        ndarray of allowed distortions (#epsilon)
#    proj_dims
#        ndarray of M's, dimensionalities of projected space,
#        tuple for (varying N, varying V)
#    distn_N
#        (1-prob)'th percentile of distortion, for different N,
#        ndarray (#(K),#(M),#(N))
#    distn_V
#        (1-prob)'th percentile of distortion, for different V,
#        ndarray (#(K),#(M),#(V))
#    """
    # separate scans for N and V
#    (M_num_N, M_num_V,
#     dist_N, dist_V, vols) = get_num_sep(param_ranges, uni_opts, mfld_info)
#
#    np.savez_compressed(filename + '.npz', num_N=M_num_N, num_V=M_num_V,
#                        dist_N=dist_N, dist_V=dist_V, vols=vols,
#                        ambient_dims=param_ranges['N'],
#                        epsilons=param_ranges['eps'],
#                        proj_dims=param_ranges['M'],
#                        prob=uni_opts['prob'])
    # alternative, double scan
    M_num, dist, vols = get_num_cmb(param_ranges, uni_opts, mfld_info)
    np.savez_compressed(filename + '.npz',
                        M_num=M_num, dist=dist, vols=vols,
                        prob=uni_opts['prob'],
                        ambient_dims=param_ranges['N'],
                        epsilons=param_ranges['eps'],
                        proj_dims=param_ranges['M'])

# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
