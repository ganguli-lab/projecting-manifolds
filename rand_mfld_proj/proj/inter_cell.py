# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:12:53 2016

@author: Subhy

Compute disortion of vectors between cell centres and vectors between edges of
balls that enclose cells, to test assertion that:

.. math::
    D_A(x) < E_C(\\epsilon,\\theta_C) \\implies D_A(y) < \\epsilon
                                            \\;\\forall y \\in C
| where C = chordal cone,
| :math:`\\theta_C` = angle between centre and edge,
| x = central vector of cone.

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
from typing import Tuple
import numpy as np
from numpy.linalg import norm
from ..iter_tricks import dbatch, denumerate
from ..myarray import array, wrap_one

# =============================================================================
# generate vectors
# =============================================================================


@wrap_one
def make_x(*siz: int)-> array:  # vector between cell centres
    """
    Generate vector between cell centers.

    Sets norm to 1, wlg

    Parameters
    ==========
    num_reps
        R, number of repetitions
    ambient_dim
        N, dimensionality of ambient space

    Returns
    =======
    x ndarray (R,N)
        a random unit vector.
    """
    x = np.random.randn(*siz)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return x


def make_y(x: array,
           theta: float,
           *siz: int) -> array:  # vector between cell edges
    """
    Generate vector from cell center to edge of ball that encloses cell, dx

    Parameters
    ==========
    x array (R,N)
        central vector of cone (unit length)
    theta
        angle between `x` and `y`
    T, num_trials
        number of attempts to find cone vector of maximum distortion

    Returns
    =======
    y ndarray (T,R,N)
        unit vector in direction from origin to the edge of the cone

    Notes
    =====
    Assumes norm(x) = 1,
    Sets norm(y) = 1,
    and y perpendicular to (x - y cos(theta))
    ==> theta = angle between x and y
    """
    cos_theta = np.cos(theta)
    y = make_x(*siz, *x.shape)
    cos_phi = (y.r @ x.c).uc
    sin_ratio = np.sqrt((1 - cos_theta**2) / (1 - cos_phi**2))
    y *= sin_ratio
    y += (cos_theta - cos_phi * sin_ratio) * x
#    y *= cos_theta
    return y


# =============================================================================
# calculate intermediaries
# =============================================================================


def guarantee_inv(distort: float,
                  theta: float,
                  proj_dim: int,
                  ambient_dim: int) -> float:
    """maximum possible distortion

    Maximum possible distortion of y given distortion of x = `distort`
    for all y in cone of angle `theta` with x.

    Parameters
    ==========
    distort
        distortion of x
    theta
        angle between centre, x, and edge of chordal cone, y
    proj_dim
        M, dimensionality of projected space
    ambient_dim
        N, dimensionality of ambient space

    Returns
    =======
    epsilon
        maximum possible distortion of y
    """
    return distort + np.sqrt(ambient_dim / proj_dim) * np.sin(theta)


def guarantee(distort: float,
              theta: float,
              proj_dim: int,
              ambient_dim: int) -> float:
    """maximum allowed distortion

    Maximum allowed distortion of x s.t. (distortion of y < `distort`) is
    guaranteed for all y in cone of angle `theta` with x.

    Parameters
    ==========
    distort
        maximum possible distortion of y
    theta
        angle between centre, x, and edge of chordal cone, y
    proj_dim
        M, dimensionality of projected space
    ambient_dim
        N, dimensionality of ambient space

    Returns
    =======
    epsilon
        maximum allowed distortion of x
    """
    return distort - np.sqrt(ambient_dim / proj_dim) * np.sin(theta)


# =============================================================================
# calculate distortion
# =============================================================================


def distortion(vec: array,
               proj_dims: array) -> array:
    """distortion of vec under projection

    Distortion of `vec` under projection.

    Assumes projection is onto first `proj_dim` dimensions & vec is normalised

    Parameters
    ==========
    vec ndarray (dT,R,N)
        unit vector being projected
    proj_dims ndarray (#(M),R)
        M, dimensionality of projected space

    Returns
    =======
    epsilon (#(M),R)
        distortion of vec under projection
    """
    axs = tuple(range(proj_dims.ndim, proj_dims.ndim + vec.ndim - 2))
    N = vec.shape[-1]
    eps = np.empty(proj_dims.shape + vec.shape[:-1])
    for m, M in enumerate(proj_dims):
        eps[m] = np.abs(np.sqrt(N / M) * norm(vec[..., 0:M], axis=-1) - 1.)
    return np.amax(eps, axis=axs)


# =============================================================================
# the whole calculation
# =============================================================================


def comparison(reps: Tuple[int],
               theta: float,
               proj_dims: array,
               ambient_dim: int) -> (float, float, float, float):
    r"""comparison of theory and experiment

    Comparison of theory and experiment
    Compute disortion of vectors between cell centres and vectors between edges
    of balls that enclose cells, to test assertion that:

    .. math::

        D_A(x) < E_C(\epsilon,\theta) \implies D_A(y) < \epsilon
        \quad                                  \forall y \in C

    where C = chordal cone,
    with angle between centre and edge = theta,
    x = central vector of cone

    Returns
    =======
    epsx ndarray (#(M),R)
        distortion of x
    gnt ndarray (#(M),R)
        guarantee(maximum distortion of y) for y in chordal cone
    epsy ndarray (#(M),R)
        maximum distortion of y for y in chordal cone
    gnti ndarray (#(M),R)
        guarantee(gnti) = distortion of x

    Parameters
    ==========
    reps (num_trials, batch_trials, num_reps)
        num_trials
            T, number of comparisons to find maximum distortion
        batch_trials
            dT, size of chunks to perform trials into
        num_reps
            R, number of times to repeat each comparison
    ambient_dim
        N, dimensionality of ambient space
    proj_dims ndarray (#(M),)
        M, dimensionality of projected space
    theta
        angle between centre and edge of chordal cone
    """
    (num_trials, batch_trials, num_reps) = reps

    x = make_x(num_reps, ambient_dim)
    epsx = distortion(x, proj_dims)
    epsy = np.zeros(proj_dims.shape + (num_reps,))

    for i in dbatch('trial', 0, num_trials, batch_trials):
        y = make_y(x, theta, batch_trials)
        np.maximum(epsy, distortion(y, proj_dims), out=epsy)

    gnt = guarantee(epsy, theta, proj_dims.c, ambient_dim)
    gnti = guarantee_inv(epsx, theta, proj_dims[..., None], ambient_dim)

    return epsx, gnt, epsy, gnti


def generate_data(reps: Tuple[int],
                  ambient_dim: int,
                  thetas: array,
                  proj_dims: array):
    r"""generate all data for plots

    Generate all data for plots and legend
    Compute disortion of vectors between cell centres and vectors between edges
    of balls that enclose cells, to test assertion that:

    .. math::

        D_A(x) < E_C(\epsilon,\theta) \implies D_A(y) < \epsilon
        \quad                                  \forall y \in C

    where C = chordal cone, with angle between centre and edge = theta
    x = central vector of cone

    Returns
    =======
    epsx ndarray (#(th),#(M),R)
        distortion of x
    gnt ndarray (#(th),#(M),R)
        guarantee(maximum distortion of y) for y in chordal cone
    epsy ndarray (#(th),#(M),R)
        maximum distortion of y for y in chordal cone
    gnti ndarray (#(th),#(M),R)
        guarantee(gnti) = distortion of x
    leg
        legend text associated with corresponding datum

    Parameters
    ==========
    reps (num_trials, batch_trials, num_reps)
        num_trials
            T, number of comparisons to find maximum distortion
        batch_trials
            dT, size of chunks to perform trials into
        num_reps
            R, number of times to repeat each comparison
    ambient_dim
        N, dimensionality of ambient space
    proj_dims ndarray (#(M),)
        M, array of dimensionalities of projected space
    thetas ndarray (#(th),)
        array of angles between centre and edge of chordal cone
    """
    epsx = np.zeros((len(thetas), len(proj_dims), reps[2]))
    gnt = np.zeros((len(thetas), len(proj_dims), reps[2]))
    epsy = np.zeros((len(thetas), len(proj_dims), reps[2]))
    gnti = np.zeros((len(thetas), len(proj_dims), reps[2]))
    leg = []

    for t, theta in denumerate('theta', thetas):
        (epsx[t], gnt[t],
         epsy[t], gnti[t]) = comparison(reps, theta, proj_dims, ambient_dim)
        for m in range(len(proj_dims)):
            leg.append(leg_text(t, m, thetas, proj_dims))
        # extra element at end of each row: label with value of theta
        leg.append(leg_text(t, len(proj_dims), thetas, proj_dims))

    return epsx, gnt, epsy, gnti, leg


# =============================================================================
# plotting
# =============================================================================


def leg_text(t: int,
             m: int,
             thetas: array,
             proj_dims: array) -> str:
    """
    Generate legend text

    labels with value of M,
    except for extra element at end of each row: labels with value of theta

    Parameters
    ==========
    t, m
        indices for proj_dims, thetas of current datum
    proj_dims
        M, list of dimensionalities of projected space
    thetas
        list of angles between centre and edge of chordal cone

    Returns
    =======
    legtext
        text for legend entry
    """
    if m == len(proj_dims):
        return r'$\theta_{\mathcal{C}} = %1.3f$' % (thetas[t])
    return r'$M = %d$' % proj_dims[m]


# =============================================================================
# options
# =============================================================================


def default_options():
    """
    Default options for generating data

    Returns
    =======
    reps (num_trials, batch_trials, num_reps)
        num_trials
            T, number of comparisons to find maximum distortion
        batch_trials
            dT, size of chunks to perform trials into
        num_reps
            R, number of times to repeat each comparison
    ambient_dim
        N, dimensionality of ambient space
    thetas
        list of angles between centre and edge of chordal cone
    proj_dims
        M, list of dimensionalities of projected space
    """
    # choose parameters
    np.random.seed(0)
    # number of samples of edge of cone
    num_trials = 2000000
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
    # angle between cone centre and edge
    thetas = np.array([0.001, 0.002, 0.003, 0.004])

    return reps, ambient_dim, thetas, proj_dims


def quick_options():
    """
    Demo options for generating test data

    Returns
    =======
    reps (num_trials, batch_trials, num_reps)
        num_trials
            T, number of comparisons to find maximum distortion
        batch_trials
            dT, size of chunks to perform trials into
        num_reps
            R, number of times to repeat each comparison
    ambient_dim
        N, dimensionality of ambient space
    thetas
        list of angles between centre and edge of chordal cone
    proj_dims
        M, list of dimensionalities of projected space
    """
    # choose parameters
    np.random.seed(0)
    # number of samples of edge of cone
    num_trials = 2000
    # size of chunks to perform trials into
    batch_trials = 100
    # number of times to repeat each comparison
    num_reps = 3
    # combine prev 3
    reps = (num_trials, batch_trials, num_reps)
    # dimensionality of ambient space
    ambient_dim = 500
    # dimensionality of projection
    proj_dims = np.array([25, 50, 75]).view(array)
    # angle between cone centre and edge
    thetas = np.array([0.001, 0.002, 0.003]).view(array)

    return reps, ambient_dim, thetas, proj_dims


# =============================================================================
# running code
# =============================================================================


def make_and_save(filename: str,
                  reps: Tuple[int],
                  ambient_dim: int,
                  thetas: array,
                  proj_dims: array):
    """
    Generate data and save in .npz file

    Parameters
    ==========
    filename
        name of .npz file, w/o extension, for data
    reps (num_trials, batch_trials, num_reps)
        num_trials
            T, number of comparisons to find maximum distortion
        batch_trials
            dT, size of chunks to perform trials into
        num_reps
            R, number of times to repeat each comparison
    ambient_dim
        N, dimensionality of ambient space
    thetas
        list of angles between centre and edge of chordal cone
    proj_dims
        M, list of dimensionalities of projected space
    """
    epsx, gnt, epsy, gnti, leg = generate_data(reps, ambient_dim, thetas,
                                               proj_dims)
    np.savez_compressed(filename + '.npz', epsx=epsx, gnt=gnt, epsy=epsy,
                        gnti=gnti, leg=leg)


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
