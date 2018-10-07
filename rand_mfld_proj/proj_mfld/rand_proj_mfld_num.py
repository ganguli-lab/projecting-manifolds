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
import numpy as np

from ..mfld import gauss_mfld as gm
from ..iter_tricks import dcontext, rdenumerate
from . import rand_proj_mfld_calc as rc
from . import rand_proj_mfld_util as ru


# K hard coded in: Options
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
    kvecs = gm.spatial_freq(mfld_info['L'], mfld_info['num'], expand)
    # Fourier transform of embedding functions, (N,Lx,Ly/2)
    embed_ft = gm.random_embed_ft(ambient_dim, kvecs, mfld_info['lambda'])
    # Fourier transform back to real space (N,Lx,Ly)
    emb = gm.embed(embed_ft)
    # find the image of the gauss map (push forward vielbein)
    grad = gm.embed_grad(embed_ft, kvecs)
    # which elements to remove to select central region
    remove = [(expand - 1) * lng // 2 for lng in mfld_info['num']]
    keep = tuple(slice(rm, -rm) for rm in remove)
    # throw out side regions, to lessen effects of periodicity
    mfld = emb[keep]
    tang = grad[keep]
    return mfld, tang


# =============================================================================
# %%* distortion calculations
# =============================================================================


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
        (1-prob)'th percentile of distortion, ndarray  (#(K),#(V),#(M))

    Returns
    -------
    M
        required projection dimensionality, ndarray (#(K),#(e),#(V))
    """

    # make sure it is strictly decreasing wrt M
    decr_eps = np.minimum.accumulate(distortions, axis=-1)
    deps = np.cumsum((np.diff(decr_eps, axis=-1) >= 0.) * 1.0e-6, axis=-1)
    decr_eps -= np.pad(deps, ((0, 0), (0, 0), (1, 0)), 'constant')

#    def func(x): return np.interp(-np.asarray(epsilon), -x, proj_dims)
#    # linearly interpolate over epsilon to find M (need - so it increases)

    def func(distn):
        """linearly interpolate over vector 1/epsilon**2 to find M"""
        return np.interp(np.asarray(epsilon)**-2, distn**-2, proj_dims)

    # apply func to M axis
    return np.apply_along_axis(func, -1, decr_eps).swapaxes(1, 2)


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
        ndarray (#(K),#(V),#(M))
    """
    Ms = param_ranges['M'][param_ranges['M'] <= mfld.shape[-1]]
    with dcontext('inds'):
        # indices for regions we keep
        region_inds = rc.region_inds_list(mfld.shape[:-1], param_ranges['Vfr'])
    with dcontext('flatten'):
        # flatten location indices, put ambient index last
        mfld2 = mfld.reshape((-1, mfld.shape[-1]))
        gmap2 = gmap.reshape((-1,) + gmap.shape[-2:]).swapaxes(-2, -1)

    # sample projs and compute max distortion of all chords (#(K),#(V),#(M),S)
    distortions = rc.distortion_m(mfld2, gmap2, Ms, uni_opts, region_inds)
    # find 1 - prob'th percentile, for each K,V,M
    eps = np.quantile(distortions, 1. - uni_opts['prob'], axis=-1)
    # find minimum M needed for epsilon, prob, for each K, epsilon, V
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
        ndarray (#(K),#(V),#(M),#(N))
    vols
        V^1/K, for each K, ndarray (#(K),#(V))
    """

    proj_req = np.empty((len(mfld_info['L']), len(param_ranges['eps']),
                         len(param_ranges['Vfr']), len(param_ranges['N'])))
    distn = np.empty((len(mfld_info['L']), len(param_ranges['M']),
                      len(param_ranges['Vfr']), len(param_ranges['N'])))

    max_vol = [ru.gmean(mfld_info['L'][:k]) / ru.gmean(mfld_info['lambda'][:k])
               for k in range(1, 1+len(mfld_info['L']))]
    vols = 2 * np.array(max_vol)[..., None] * param_ranges['Vfr']

    # generate manifold
    with dcontext('mfld'):
        mfld, tang = make_surf(param_ranges['N'][-1], mfld_info)

    for i, N in rdenumerate('N', param_ranges['N']):
        gmap = gm.vielbein(tang[..., :N, :])
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
                'batch': 20}

    mfld_info = {'num': (32, 32),  # number of points to sample
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

    M_num
        values of M when varying K,epsilon,V,N, ndarray (#(K),#(e),#(V),#(N))
    prob
        allowed failure probability
    ambient_dims
        ndarray of N's, dimensionality of ambient space, (#(N),)
    vols
        V^1/K, for each K, ndarray (#(K),#(V))
    epsilons
        ndarray of allowed distortions (#(e),)
    proj_dims
        ndarray of M's, dimensionalities of projected space, (#(M),)
    distn
        (1-prob)'th percentile of distortion, for different N, V, M, K
        ndarray (#(K),#(M),#(V),#(N))
    """
    if uni_opts['samples'] % uni_opts['batch'] != 0:
        msg = 'samples must be divisible by batches. samples: {}, batch: {}.'
        raise ValueError(msg.format(uni_opts['samples'], uni_opts['batch']))

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
