# -*- coding: utf-8 -*-
r"""
Created on Mon May 16 22:12:53 2016

@author: Subhy

Compute disortion of tangent space at cell centre and tangent spaces at edge of
a Grassmannian region that encloses the image of cell under the Gauss map,
to test assertion that:

.. math::
    D_A(U) < E_T(\epsilon,\theta_T) \implies D_A(U') < \epsilon
                                             \;\forall U' \in T
| where T = tangential cone,
| with principal angles between U and U' < :math:`\theta_T`,
| U = central subspace.

Functions
=========
generate_data
    calculate all numeric quantities
default_options
    default options for long numerics for paper
quick_options
    default options for quick numerics for demo
make_and_save
    generate data and save npz file
"""
from typing import Sequence, Tuple
import numpy as np
from numpy.linalg import svd
from ..iter_tricks import dbatch, denumerate
from ..myarray import larray, wrap_one

# =============================================================================
# generate vectors
# =============================================================================


@wrap_one
def make_basis(*siz: int) -> larray:
    """
    Generate orthonormal basis for central subspace

    Returns
    -------
    U ndarray (R,N,K)
        basis of subspace

    Parameters
    ----------
    count
        R, # bases to generate
    ambient_dim
        N, dimensionality of ambient space
    sub_dim
        K, dimensionality of tangent subspace
    """
    if len(siz) <= 1:
        U = np.random.randn(*siz)
        U = np.linalg.qr(U)[0]
        return U

    spaces = np.random.randn(*siz)
    U = np.empty(siz)
    for ii in np.ndindex(siz[:-2]):
        # orthogonalise with Gram-Schmidt
        U[ii] = np.linalg.qr(spaces[ii])[0]
    return U


def make_basis_perp(ambient_dim: int, sub_dim: int,
                    *count: int) -> (larray, larray):
    """
    Generate orthonormal basis for central subspace and its orthogonal
    complement

    Returns
    -------
    U_par ndarray (R,N,K)
        basis of subspace
    U_perp ndarray (R,N,N-K)
        basis of orthogonal complement of subspace

    Parameters
    ----------
    ambient_dim
        N, dimensionality of ambient space
    sub_dim
        K, dimensionality of tangent subspace
    count
        R, # bases to generate
    """
    U = make_basis(*count, ambient_dim, ambient_dim)
    return U[..., 0:sub_dim], U[..., sub_dim:]


def make_basis_other(U_par: larray,
                     U_perp: larray,
                     theta_max: float,
                     *num_trials: int) -> larray:
    """
    Generate orthonormal basis for another subspace on edge of cone T

    Returns
    -------
    U' ndarray (T,R,N,K)
        basis of subspace on edge of T

    Parameters
    ----------
    U_par ndarray (R,N,K)
        basis of subspace
    U_perp ndarray (R,N,N-K)
        basis of orthogonal complement of subspace
    theta_max
        max principal angle between U_par and U'
    num_trials
        T, # bases to generate

    Notes
    -----
    Most general U' w/ principal angles :math:`\\theta_a` is:

    .. math::
        U' = (U_\\parallel S_\\parallel \\cos\\Theta
             + U_\\perp S_\\perp \\sin\\Theta) R'
    where:

    | :math:`L = min(K,N-K)`
    | :math:`S_\\parallel,R`: KxL & :math:`S_\\parallel'S_\\parallel = R'R = I`
    | :math:`S_\\perp`: (N-K)xL and :math:`S_\\perp' S_\\perp = I`
    | :math:`\\Theta = diag(\\theta_a)`

    We set :math:`\\theta_1 = \\theta_\\max` and independently sample
    :math:`\\theta_{a>1}` uniformly in `[0,\\theta_\\max]`
    (not the Harr measure)
    """
    m = min(U_par.shape[-1], U_perp.shape[-1])
    count = num_trials + U_par.shape[:-2]
    theta = np.random.rand(*count, 1, m)
    theta[..., 0] = 1.
    theta *= theta_max
    costh = np.cos(theta)
    sinth = np.sin(theta)

    S_par = make_basis(*count, U_par.shape[-1], m)
    S_perp = make_basis(*count, U_perp.shape[-1], m)
    R = make_basis(*count, U_par.shape[-1], m)

    return (U_par @ S_par * costh + U_perp @ S_perp * sinth) @ R.t


# =============================================================================
# calculate intermediaries
# =============================================================================


def guarantee_inv(distort: float,
                  theta: float,
                  proj_dim: int,
                  ambient_dim: int) -> float:
    """maximum possible distortion

    Maximum possible distortion of U' given distortion of U = `distort`
    for all U' in tangential cone of max principal angle `theta` with U.

    Parameters
    ----------
    distort
        distortion of U
    ambient_dim
        N, dimensionality of ambient space
    proj_dim
        M, dimensionality of projected space
    """
    return distort + (ambient_dim / proj_dim) * np.sin(theta)


def guarantee(distort: float,
              theta: float,
              proj_dim: int,
              ambient_dim: int) -> float:
    """maximum allowed distortion

    Maximum allowed distortion of U s.t. (distortion of U' < `distort`)
    guaranteed for all U' in tangential cone of max principal angle `theta`
    with U.

    Parameters
    ----------
    distort
        distortion of U'
    ambient_dim
        N, dimensionality of ambient space
    proj_dim
        M, dimensionality of projected space
    """
    return distort - (ambient_dim / proj_dim) * np.sin(theta)


def max_pang(U1, U2):  # sine of largest principal angle between spaces
    """
    Sine of largest principal angle between spaces spanned bu `U1` and `U2`

    Parameters
    ----------
    U!, U2 ndarray (R,N,K), (T,R,N,K)
        basis of subspace

    Returns
    -------
    sin(theta) ndarray (T,R)
    """
    gram = U1.T @ U2  # (T,R,K,K)
    sv = svd(gram, compute_uv=False)  # (T,R,K)
    sines = np.sqrt(1. - sv**2)
    return np.amax(sines, axis=-1)  # (T,R)

# =============================================================================
# calculate distortion
# =============================================================================


def distortion(space: larray,
               proj_dims: larray) -> float:
    """distortion of vec under projection

    Distortion of subspace under projection.

    Assumes projection is onto first `proj_dim` dimensions

    Parameters
    ----------
    space ndarray (dT,R,N,K)
        orthonormal basis for subspace
    proj_dims ndarray (#(M),)
        M, dimensionality of projected space

    Returns
    -------
    eps ndarray (#(M),R)
     """
    axs = tuple(range(proj_dims.ndim, proj_dims.ndim + space.ndim - 3)) + (-1,)
    N = space.shape[-2]
    dist = np.empty(proj_dims.shape + space.shape[:-2] + space.shape[-1:])
    for m, M in enumerate(proj_dims):
        sv = svd(space[..., 0:M, :], compute_uv=False)  # (#(M),dT,R,K)
        dist[m] = np.abs(np.sqrt(N / M) * sv - 1.)  # (#(M),dT,R,K)
    return np.amax(dist, axis=axs)  # (#(M),R)


# =============================================================================
# the whole calculation
# =============================================================================


def comparison(reps: Sequence[int],
               theta: float,
               sub_dim: int,
               proj_dims: larray,
               ambient_dim: int) -> (float, float, float, float):
    """comparison of theory and experiment

    Comparison of theory and experiment
    Compute disortion of central subspace and subspaces at edges of cone that
    encloses the image of cell under the Gauss map, to test assertion that:

    .. math::
        D_A(U) < E_T(\\epsilon,\\theta) \\implies D_A(U') < \\epsilon
                                               \\;\\forall U' \\in T
    | where T = tangential cone,
    | :math:`\\theta` = max principal angle between centre and edge,
    | U = central subspace of cone

    Returns
    -------
    epsilon ndarray (#(M),R)
        distortion of central subspace U
    gnt ndarray (#(M),R)
        guarantee(maximum distortion of U' for U' in tangential cone
    epsilonb ndarray (#(M),R)
        maximum distortion of U' for U' in tangential cone
    gnti ndarray (#(M),R)
        guarantee(gnti) = distortion of central subspace U

    Parameters
    ----------
    reps
        num_trials
            T, number of comparisons to find maximum distortion
        batch_trials
            dT, size of chunks to perform trials into
        num_reps
            R, number of times to repeat each comparison
    theta
        max principal angle between centre and edge of chordal cone
    sub_dim
        K, dimensionality of subspace
    proj_dims ndarray (#(M),)
        M, dimensionality of projected space
    ambient_dim
        N, dimensionality of ambient space
    """
    (num_trials, batch_trials, num_reps) = reps

    U_par, U_perp = make_basis_perp(ambient_dim, sub_dim, num_reps)
    epsilon = distortion(U_par, proj_dims)  # (#(M),R)
    epsilonb = np.zeros_like(epsilon)  # (#(M),R)

    for i in dbatch('trial', 0, num_trials, batch_trials):
        U2 = make_basis_other(U_par, U_perp, theta, batch_trials)
        np.maximum(epsilonb, distortion(U2, proj_dims), out=epsilonb)

    gnt = guarantee(epsilonb, theta, proj_dims[..., None], ambient_dim)
    gnti = guarantee_inv(epsilon, theta, proj_dims[..., None], ambient_dim)

    return epsilon, gnt, epsilonb, gnti


def generate_data(reps: Sequence[int],
                  amb_dim: int,
                  thetas: larray,
                  sub_dims: larray,
                  proj_dims: larray) -> (np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray):
    """
    Generate all data for plots and legend
    Compute disortion of central subspace and subspaces at edges of cone that
    encloses the image of cell under the Gauss map, to test assertion that:

    .. math::
        D_A(U) < E_T(\\epsilon,\\theta) \\implies D_A(U') < \\epsilon
                                               \\;\\forall U' \\in T
    | where T = tangential cone,
    | :math:`\\theta` = max principal angle between centre and edge,
    | U = central subspace of cone

    Returns
    -------
    eps ndarray (#(th),#(K),#(M),R)
        distortion of central subspace U
    gnt ndarray (#(th),#(K),#(M),R)
        guarantee(maximum distortion of U' for U' in tangential cone
    epsb ndarray (#(th),#(K),#(M),R)
        maximum distortion of U' for U' in tangential cone
    gnti ndarray (#(th),#(K),#(M),R)
        guarantee(gnti) = distortion of central subspace U
    leg
        legend text associated with corresponding datum

    Parameters
    ----------
    reps
        num_trials
            T, number of comparisons to find maximum distortion
        batch_trials
            dT, size of chunks to perform trials into
        num_reps
            R, number of times to repeat each comparison
    amb_dim
        N, dimensionality of ambient space
    thetas ndarray (#(th),)
        list of angles between centre and edge of chordal cone
    sub_dims ndarray (#(K),)
        K, list of dimensionalities of subspace
    proj_dims ndarray (#(M),)
        M, set of dimensionalities of projected space
    """
    eps = np.zeros((len(thetas), len(sub_dims), len(proj_dims), reps[2]))
    gnt = np.zeros((len(thetas), len(sub_dims), len(proj_dims), reps[2]))
    epsb = np.zeros((len(thetas), len(sub_dims), len(proj_dims), reps[2]))
    gnti = np.zeros((len(thetas), len(sub_dims), len(proj_dims), reps[2]))
    leg = []

    for t, theta in denumerate('theta', thetas):
        for k, K in denumerate('K', sub_dims):
            ind = (t, k)
            (eps[ind],
             gnt[ind],
             epsb[ind],
             gnti[ind]) = comparison(reps, theta, K, proj_dims, amb_dim)
            for m in range(len(proj_dims)):
                leg.append(leg_text(t, k, m, thetas, sub_dims, proj_dims))
            # extra element at end of each row: label with value of M
            leg.append(leg_text(t, k, len(proj_dims), thetas, sub_dims,
                                proj_dims))
        # extra element at end of each column: label with value of theta
        leg.append(leg_text(t, len(sub_dims), len(proj_dims), thetas, sub_dims,
                            proj_dims))

    return eps, gnt, epsb, gnti, leg


# =============================================================================
# plotting
# =============================================================================


def leg_text(t: int, k: int, m: int,
             thetas: larray,
             sub_dims: larray,
             proj_dims: larray) -> Sequence[str]:
    """
    Generate legend text

    labels with value of theta,
    except for extra element at end of each row: labels with value of M,
    except for extra element at end of each column: labels with value of K

    Returns
    -------
    legtext
        list of strings containing legend entries

    Parameters
    ----------
    t, k, m
        indices of currrent datum (theta, K, M)
    thetas
        list of angles between centre and edge of chordal cone
    sub_dims
        K, list of dimensionalities of subspace
    proj_dims
        M, set of dimensionalities of projected space
    """
    if k == len(sub_dims):
        legtext = r'$\theta_{\mathcal{T}} = %1.3f$' % thetas[t]
    elif m == len(proj_dims):
        legtext = r'$K = %d$' % sub_dims[k]
    else:
        legtext = r'$M = %d$' % proj_dims[m]
    return legtext


# =============================================================================
# options
# =============================================================================


def default_options() -> (Tuple[int], int,
                          larray, larray, larray):
    """
    Default options for generating data

    Returns
    -------
    reps
        num_trials
            number of comparisons to find maximum distortion
        batch_trials
            size of chunks to perform trials into
        num_reps
            number of times to repeat each comparison
    ambient_dim
        N, dimensionality of ambient space
    thetas
        list of angles between centre and edge of chordal cone
    proj_dims
        M, set of dimensionalities of projected space
    sub_dims
        K, list of dimensionalities of subspace
    """
    # choose parameters
    np.random.seed(0)
    # number of samples of edge of cone
    num_trials = 200000
    # size of chunks to perform trials into
    batch_trials = 100
    # number of times to repeat each comparison
    num_reps = 5
    # combine prev 3
    reps = (num_trials, batch_trials, num_reps)
    # dimensionality of ambient space
    ambient_dim = 1000
    # dimensionality of projection
    proj_dims = np.array([50, 75, 100])
    # dimensionality of subspace
    sub_dims = np.array([5, 10])
    # max angle between cone centre and edge
    thetas = np.array([0.001, 0.002, 0.003, 0.004])

    return reps, ambient_dim, thetas, proj_dims, sub_dims


def quick_options() -> (Tuple[int], int,
                        Sequence[float], Sequence[int], Sequence[int]):
    """
    Demo options for generating test data

    Returns
    -------
    reps
        num_trials
            number of comparisons to find maximum distortion
        batch_trials
            size of chunks to perform trials into
        num_reps
            number of times to repeat each comparison
    ambient_dim
        N, dimensionality of ambient space
    thetas
        list of angles between centre and edge of chordal cone
    proj_dims
        M, set of dimensionalities of projected space
    sub_dims
        K, list of dimensionalities of subspace
    """
    # choose parameters
    np.random.seed(0)
    # number of samples of edge of cone
    num_trials = 2000
    # size of chunks to split trials
    batch_trials = 100
    # number of times to repeat each comparison
    num_reps = 3
    # combine prev 3
    reps = (num_trials, batch_trials, num_reps)
    # dimensionality of ambient space
    ambient_dim = 500
    # dimensionality of projection
    proj_dims = np.array([25, 50, 75])
    # dimensionality of subspace
    sub_dims = np.array([5, 10])
    # max angle between cone centre and edge
    thetas = np.array([0.001, 0.002, 0.003])

    return reps, ambient_dim, thetas, proj_dims, sub_dims


# =============================================================================
# running code
# =============================================================================


def make_and_save(filename: str,
                  reps: Sequence[int],
                  ambient_dim: int,
                  thetas: Sequence[float],
                  proj_dims: Sequence[int],
                  sub_dims: Sequence[int]):  # generate data and save
    """
    Generate data and save in .npz file

    Parameters
    ----------
    filename
        name of .npz file, w/o extension, for data
    reps
        num_trials
            number of comparisons to find maximum distortion
        batch_trials
            size of chunks to perform trials into
        num_reps
            number of times to repeat each comparison
    ambient_dim
        N, dimensionality of ambient space
    thetas
        list of angles between centre and edge of chordal cone
    proj_dims
        M, set of dimensionalities of projected space
    sub_dims
        K, list of dimensionalities of subspace
    """
    eps, gnt, epsb, gnti, leg = generate_data(reps, ambient_dim, thetas,
                                              sub_dims, proj_dims)
    np.savez_compressed(filename + '.npz', eps=eps, gnt=gnt, epsb=epsb,
                        gnti=gnti, leg=leg)


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
