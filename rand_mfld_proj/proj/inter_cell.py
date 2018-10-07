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
from typing import Sequence
import numpy as np
from ..iter_tricks import dbatch, denumerate
# =============================================================================
# generate vectors
# =============================================================================


def make_x(*siz: int)-> np.ndarray:  # vector between cell centres
    """
    Generate vector between cell centers.

    Sets norm to 1, wlg

    Parameters
    ==========
    ambient_dim
        N, dimensionality of ambient space

    Returns
    =======
    x
        a random unit vector.
    """
    x = np.random.randn(*siz)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return x


def make_y(x: float,
           theta: float,
           *siz: int) -> np.ndarray:  # vector between cell edges
    """
    Generate vector from cell center to edge of ball that encloses cell, dx

    Parameters
    ==========
    x
        central vector of cone
    theta
        angle between `x` and `y`

    Returns
    =======
    y
        vector from origin to the edge of the cone

    Notes
    =====
    Assumes norm(x) = 1,
    Sets norm(y) = cos(theta),
    and y perpendicular to (x - y)
    ==> theta = angle between x and y
    """
    cos_theta = np.cos(theta)
    y = make_x(*siz, *x.shape)
    cos_phi = (y[..., None, :] @ x[..., None]).squeeze(-1)
    sin_ratio = np.sqrt((1 - cos_theta**2) / (1 - cos_phi**2))
    y *= sin_ratio
    y += (cos_theta - cos_phi * sin_ratio) * x
    y *= cos_theta
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


def distortion(vec: np.ndarray,
               proj_dim: int) -> float:
    """distortion of vec under projection

    Distortion of vec under projection.

    Assumes projection is onto first `proj_dim` dimensions

    Parameters
    ==========
    vec
        vector being projected, (N,)
    proj_dim
        M, dimensionality of projected space

    Returns
    =======
    epsilon
        distortion of vec under projection
    """
    axes = tuple(range(vec.ndim - 2)) + (-1,)
    eps = np.abs(np.sqrt(vec.shape[-1] / proj_dim) *
                 np.linalg.norm(vec[..., 0:proj_dim], axis=-1) /
                 np.linalg.norm(vec, axis=-1) - 1.)
    return np.amax(eps, axis=axes)


# =============================================================================
# the whole calculation
# =============================================================================


def comparison(reps: Sequence[int],
               theta: float,
               proj_dim: int,
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
    epsx
        distortion of x
    gnt
        guarantee(maximum distortion of y) for y in chordal cone
    epsy
        maximum distortion of y for y in chordal cone
    gnti
        guarantee(gnti) = distortion of x

    Parameters
    ==========
    reps (num_trials, batch_trials, num_reps)
        num_trials
            number of comparisons to find maximum distortion
        batch_trials
            size of chunks to perform trials into
        num_reps
            number of times to repeat each comparison
    ambient_dim
        N, dimensionality of ambient space
    proj_dim
        M, dimensionality of projected space
    theta
        angle between centre and edge of chordal cone
    """
    x = make_x(reps[2], ambient_dim)

    epsx = distortion(x, proj_dim)
    epsy = np.zeros(reps[2])

    for i in dbatch('trial', 0, *reps[:2]):
        y = make_y(x, theta, reps[1])
        np.maximum(epsy, distortion(y, proj_dim), out=epsy)

    gnt = guarantee(epsy, theta, proj_dim, ambient_dim)
    gnti = guarantee_inv(epsx, theta, proj_dim, ambient_dim)

    return epsx, gnt, epsy, gnti


def generate_data(reps: Sequence[int],
                  ambient_dim: int,
                  thetas: Sequence[float],
                  proj_dims: Sequence[int]):
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
    epsx
        distortion of x
    gnt
        guarantee(maximum distortion of y) for y in chordal cone
    epsy
        maximum distortion of y for y in chordal cone
    gnti
        guarantee(gnti) = distortion of x
    leg
        legend text associated with corresponding datum

    Parameters
    ==========
    reps (num_trials, batch_trials, num_reps)
        num_trials
            number of comparisons to find maximum distortion
        batch_trials
            size of chunks to perform trials into
        num_reps
            number of times to repeat each comparison
    ambient_dim
        N, dimensionality of ambient space
    proj_dims
        M, list of dimensionalities of projected space
    thetas
        list of angles between centre and edge of chordal cone
    num_reps
        number of times to repeat each comparison
    """
    epsx = np.zeros((len(thetas), len(proj_dims), reps[2]))
    gnt = np.zeros((len(thetas), len(proj_dims), reps[2]))
    epsy = np.zeros((len(thetas), len(proj_dims), reps[2]))
    gnti = np.zeros((len(thetas), len(proj_dims), reps[2]))
    leg = []

    for i, theta in denumerate('theta', thetas):
        for j, M in denumerate('M', proj_dims):
            (epsx[i, j],
             gnt[i, j],
             epsy[i, j],
             gnti[i, j]) = comparison(reps, theta, M, ambient_dim)
            leg.append(leg_text(i, j, thetas, proj_dims))
        # extra element at end of each row: label with value of theta
        leg.append(leg_text(i, len(proj_dims), thetas, proj_dims))

    return epsx, gnt, epsy, gnti, leg


# =============================================================================
# plotting
# =============================================================================


def leg_text(i: int,
             j: int,
             thetas: Sequence[float],
             proj_dims: Sequence[int]) -> str:
    """
    Generate legend text

    labels with value of M,
    except for extra element at end of each row: labels with value of theta

    Parameters
    ==========
    i, j
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
    if j == len(proj_dims):
        legtext = r'$\theta_{\mathcal{C}} = %1.3f$' % (thetas[i])
    else:
        legtext = r'$M = %d$' % proj_dims[j]
    return legtext


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
    proj_dims = [50, 75, 100]
    # angle between cone centre and edge
    thetas = [0.001, 0.002, 0.003, 0.004]

    return reps, ambient_dim, thetas, proj_dims


def quick_options():
    """
    Demo options for generating test data

    Returns
    =======
    reps (num_trials, batch_trials, num_reps)
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
    proj_dims = [25, 50, 75]
    # angle between cone centre and edge
    thetas = [0.001, 0.002, 0.003]

    return reps, ambient_dim, thetas, proj_dims


# =============================================================================
# running code
# =============================================================================


def make_and_save(filename: str,
                  reps: Sequence[int],
                  ambient_dim: int,
                  thetas: Sequence[float],
                  proj_dims: Sequence[int]):
    """
    Generate data and save in .npz file

    Parameters
    ==========
    filename
        name of .npz file, w/o extension, for data
    reps (num_trials, batch_trials, num_reps)
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
