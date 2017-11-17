# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:12:53 2016

@author: Subhy

Compute disortion of tangent space at cell centre and tangent spaces at edge of
a Grassmannian region that encloses the image of cell under the Gauss map,
to test assertion that:
    D_A(U) < E_T(\epsilon,\theta_T) ==> D_A(U') < \epsilon \forall U' in T
    where T = tangential cone,
    with principal angles between U and U' < \theta_T,
    U = central subspace

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
import numpy as np
from ..disp_counter import disp_enum
from ..disp_counter import denum
from ..disp_counter import display_counter as disp

# =============================================================================
# generate vectors
# =============================================================================


def make_basis(ambient_dim, sub_dim):  # generate basis for 1st space
    """
    Generate orthonormal basis for central subspace and its orthogonal
    complement

    Returns
    -------
    U_par
        basis of subspace
    U_perp
        basis of orthogonal complement of subspace

    Parameters
    ----------
    ambient_dim
        N, dimensionality of ambient space
    sub_dim
        K, dimensionality of tangent subspace
    """
    U = np.random.randn(ambient_dim, ambient_dim)
    U = np.linalg.qr(U)[0]
    return U[:, 0:sub_dim], U[:, sub_dim:]


def make_basis_other(U_par, U_perp, theta_max):  # generate basis for 2nd space
    """
    Generate orthonormal basis for another subspace on edge of cone T

    Returns
    -------
    U'
        basis of subspace on edge of T

    Parameters
    ----------
    U_par
        basis of subspace
    U_perp
        basis of orthogonal complement of subspace
    theta_max
        max principal angle between U_par and U'

    Notes
    -----
    Most general U' w/ principal angles theta_a is:
    U' = (U_par S_par cos(Theta) + U_perp S_perp sin(Theta)) R
    where:
    S_par, R: KxK and S_par' S_par = R' R = I
    S_perp: (N-K)xK and S_perp' S_perp = I
    Theta = diag(theta_a)
    We set theta_1 = theta_max and independently sample theta_a > 1 uniformly
    in [0,theta_max] (not the Harr measure)
    """
    theta = np.random.randn(U_par.shape[1])
    theta[0] = 1.
    theta *= theta_max
    costh = np.diag(np.cos(theta))
    sinth = np.diag(np.sin(theta))

    S_par = np.random.randn(U_par.shape[1], U_par.shape[1])
    S_par = np.linalg.qr(S_par)[0]
    S_perp = np.random.randn(U_perp.shape[1], U_par.shape[1])
    S_perp = np.linalg.qr(S_perp)[0]

    R = np.random.randn(U_par.shape[1], U_par.shape[1])
    R = np.linalg.qr(R)[0]

    return (U_par @ S_par @ costh + U_perp @ S_perp @ sinth) @ R


# =============================================================================
# calculate intermediaries
# =============================================================================


def guarantee_inv(distort, theta, proj_dim, ambient_dim):
    """maximum allowed distortion

    Maximum possible distortion of U' given distortion of U = distort
    for all U' in tangential cone of max principal angle theta with U.

    Parameters
    ----------
    ambient_dim
        N, dimensionality of ambient space
    proj_dim
        M, dimensionality of projected space
    """
    return distort + (ambient_dim / proj_dim) * np.sin(theta)


def guarantee(distort, theta, proj_dim, ambient_dim):
    """maximum allowed distortion

    Maximum allowed distortion of U s.t. distortion of U' < distort guaranteed
    for all U' in tangential cone of max principal angle theta with U.

    Parameters
    ----------
    ambient_dim
        N, dimensionality of ambient space
    proj_dim
        M, dimensionality of projected space
    """
    return distort - (ambient_dim / proj_dim) * np.sin(theta)


def max_pang(U1, U2):  # sine of largest principal angle between spaces
    """
    Sine of largest principal angle between spaces spanned bu U1 and U2
    """
    gram = U1.T @ U2
    sv = np.linalg.svd(gram)[1]
    sines = np.sqrt(1. - sv**2)
    return np.amax(sines)

# =============================================================================
# calculate distortion
# =============================================================================


def distortion(space, proj_dim, ambient_dim):
    """distortion of vec under projection

    Distortion of subspace under projection.

    Assumes projection is onto first proj_dim dimensions

    Parameters
    ----------
    space
        orthonormal basis for subspace
    ambient_dim
        N, dimensionality of ambient space
    proj_dim
        M, dimensionality of projected space
     """
    sv = np.linalg.svd(space[0:proj_dim, :])[1]
    return np.amax(np.abs(np.sqrt(ambient_dim / proj_dim) * sv - 1.))


# =============================================================================
# the whole calculation
# =============================================================================


def comparison(num_trials, theta, proj_dim, sub_dim, ambient_dim):
    """comparison of theory and experiment

    Comparison of theory and experiment
    Compute disortion of central subspace and subspaces at edges of cone that
    encloses the image of cell under the Gauss map, to test assertion that:
        D_A(U) < E_T(epsilon,theta) ==> D_A(U') < epsilon for all U' in T
    where T = tangential cone,
    with max principal angle between centre and edge = theta,
    U = central subspace of cone

    Returns
    -------
    epsilon
        distortion of central subspace U
    gnt
        guarantee(maximum distortion of U' for U' in tangential cone
    epsilonb
        maximum distortion of U' for U' in tangential cone
    gnti
        guarantee(gnti) = distortion of central subspace U

    Parameters
    ----------
    num_trials
        number of comparisons to find maximum distortion
    ambient_dim
        N, dimensionality of ambient space
    proj_dim
        M, dimensionality of projected space
    sub_dim
        K, dimensionality of subspace
    theta
        max principal angle between centre and edge of chordal cone
    """
    U_par, U_perp = make_basis(ambient_dim, sub_dim)
    epsilon = distortion(U_par, proj_dim, ambient_dim)
    epsilonb = 0.

    for i in disp('trial', num_trials):
        U2 = make_basis_other(U_par, U_perp, theta)
        epsilonb = np.maximum(epsilonb, distortion(U2, proj_dim, ambient_dim))

    gnt = guarantee(epsilonb, theta, proj_dim, ambient_dim)
    gnti = guarantee_inv(epsilon, theta, proj_dim, ambient_dim)

    return epsilon, gnt, epsilonb, gnti


def generate_data(num_trials, ambient_dim, thetas, proj_dims, sub_dims,
                  num_reps):  # gnereate all data for plots
    """
    Generate all data for plots and legend
    Compute disortion of central subspace and subspaces at edges of cone that
    encloses the image of cell under the Gauss map, to test assertion that:
        D_A(U) < E_T(epsilon,theta) ==> D_A(U') < epsilon for all U' in T
    where T = tangential cone,
    with max principal angle between centre and edge = theta,
    U = central subspace of cone

    Returns
    -------
    eps
        distortion of central subspace U
    gnt
        guarantee(maximum distortion of U' for U' in tangential cone
    epsb
        maximum distortion of U' for U' in tangential cone
    gnti
        guarantee(gnti) = distortion of central subspace U
    leg
        legend text associated with corresponding datum

    Parameters
    ----------
    num_trials
        number of comparisons to find maximum distortion
    ambient_dim
        N, dimensionality of ambient space
    proj_dims
        M, set of dimensionalities of projected space
    sub_dims
        K, list of dimensionalities of subspace
    thetas
        list of angles between centre and edge of chordal cone
    num_reps
        number of times to repeat each comparison
    """
    eps = np.zeros((len(thetas), len(proj_dims), len(sub_dims), num_reps))
    gnt = np.zeros((len(thetas), len(proj_dims), len(sub_dims), num_reps))
    epsb = np.zeros((len(thetas), len(proj_dims), len(sub_dims), num_reps))
    gnti = np.zeros((len(thetas), len(proj_dims), len(sub_dims), num_reps))
    leg = []

    for i, theta in denum('theta', thetas):
        for j, M in denum('M', proj_dims):
            for k, K in denum('K', sub_dims):
                for r in disp('rep', num_reps):
                    (eps[i, j, k, r],
                     gnt[i, j, k, r],
                     epsb[i, j, k, r],
                     gnti[i, j, k, r]) = comparison(num_trials, theta, M, K,
                                                    ambient_dim)
                leg.append(leg_text(i, j, k, thetas, proj_dims, sub_dims))
            # extra element at end of each row: label with value of M
            leg.append(leg_text(i, j, len(sub_dims), thetas, proj_dims,
                                sub_dims))
        # extra element at end of each column: label with value of theta
        leg.append(leg_text(i, len(proj_dims), len(sub_dims), thetas,
                            proj_dims, sub_dims))

    return eps, gnt, epsb, gnti, leg


# =============================================================================
# plotting
# =============================================================================


def leg_text(i, j, k, thetas, proj_dims, sub_dims):  # generate legend text
    """
    Generate legend text

    labels with value of K,
    except for extra element at end of each row: labels with value of M
    except for extra element at end of each column: labels with value of theta
    """
    if j == len(proj_dims):
        legtext = r'$\theta_{\mathcal{T}} = %1.3f$' % thetas[i]
    elif k == len(sub_dims):
        legtext = r'$M = %d$' % proj_dims[j]
    else:
        legtext = r'$K = %d$' % sub_dims[k]
    return legtext


# =============================================================================
# options
# =============================================================================


def default_options():
    """
    Default options for generating data

    Returns
    -------
    num_trials
        number of comparisons to find maximum distortion
    ambient_dim
        N, dimensionality of ambient space
    thetas
        list of angles between centre and edge of chordal cone
    proj_dims
        M, set of dimensionalities of projected space
    sub_dims
        K, list of dimensionalities of subspace
    num_reps
        number of times to repeat each comparison
    """
    # choose parameters
    np.random.seed(0)
    # number of samples of edge of cone
    num_trials = 200000
    # number of times to repeat each comparison
    num_reps = 5
    # dimensionality of ambient space
    ambient_dim = 1000
    # dimensionality of projection
    proj_dims = [50, 75, 100]
    # dimensionality of subspace
    sub_dims = [5, 10]
    # max angle between cone centre and edge
    thetas = [0.001, 0.002, 0.003, 0.004]

    return num_trials, ambient_dim, thetas, proj_dims, sub_dims, num_reps


def quick_options():
    """
    Demo options for generating test data

    Returns
    -------
    num_trials
        number of comparisons to find maximum distortion
    ambient_dim
        N, dimensionality of ambient space
    thetas
        list of angles between centre and edge of chordal cone
    proj_dims
        M, set of dimensionalities of projected space
    sub_dims
        K, list of dimensionalities of subspace
    num_reps
        number of times to repeat each comparison
    """
    # choose parameters
    np.random.seed(0)
    # number of samples of edge of cone
    num_trials = 2000
    # number of times to repeat each comparison
    num_reps = 3
    # dimensionality of ambient space
    ambient_dim = 500
    # dimensionality of projection
    proj_dims = [25, 50, 75]
    # dimensionality of subspace
    sub_dims = [5, 10]
    # max angle between cone centre and edge
    thetas = [0.001, 0.002, 0.003]

    return num_trials, ambient_dim, thetas, proj_dims, sub_dims, num_reps


# =============================================================================
# running code
# =============================================================================


def make_and_save(filename, num_trials, ambient_dim, thetas, proj_dims,
                  sub_dims, num_reps):  # generate data and save
    """
    Generate data and save in .npz file

    Parameters
    ----------
    filename
        name of .npz file, w/o extension, for data
    num_trials
        number of comparisons to find maximum distortion
    ambient_dim
        N, dimensionality of ambient space
    thetas
        list of angles between centre and edge of chordal cone
    proj_dims
        M, set of dimensionalities of projected space
    sub_dims
        K, list of dimensionalities of subspace
    num_reps
        number of times to repeat each comparison
    """
    eps, gnt, epsb, gnti, leg = generate_data(num_trials, ambient_dim, thetas,
                                              proj_dims, sub_dims, num_reps)
    np.savez_compressed(filename + '.npz', eps=eps, gnt=gnt, epsb=epsb,
                        gnti=gnti, leg=leg)


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
