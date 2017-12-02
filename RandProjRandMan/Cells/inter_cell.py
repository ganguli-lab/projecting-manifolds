# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:12:53 2016

@author: Subhy

Compute disortion of vectors between cell centres and vectors between edges of
balls that enclose cells, to test assertion that:
    D_A(x) < E_C(\epsilon,\theta_C) ==> D_A(y) < \epsilon \forall y in C
    where C = chordal cone,
    with angle between centre and edge = \theta_C,
    x = central vector of cone

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
from typing import Iterable
import numpy as np
from ..disp_counter import display_counter as disp
from ..disp_counter import denum

# =============================================================================
# generate vectors
# =============================================================================


def make_x(ambient_dim: int)-> np.ndarray:  # vector between cell centres
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
    x = np.random.randn(ambient_dim)
    x /= np.sqrt(x @ x)
    return x


def make_dx(x: float,
            theta: float) -> np.ndarray:  # vector inside cell
    """
    Generate vector from cell center to edge of ball that encloses cell, dx

    Parameters
    ==========
    x
        central vector of cone
    theta
        angle between `x` and `x + dx`

    Returns
    =======
    dx
        vector from end of `x` to the edge of the cone

    Notes
    =====
    Assumes norm(x) = 1,
    Sets norm(dx) = sin(theta),
    and dx perpendicular to (x + dx)
    ==> theta = angle between x and (x + dx)
    """
    dx = np.random.randn(x.shape[0])
    dx -= x * (x @ dx)
    dx *= np.cos(theta) / np.linalg.norm(dx, axis=0)
    dx -= np.sin(theta) * x
    dx *= np.sin(theta)
    return dx


# =============================================================================
# calculate intermediaries
# =============================================================================


def guarantee_inv(distort: float,
                  theta: float,
                  proj_dim: int,
                  ambient_dim: int) -> float:
    """maximum possible distortion

    Maximum possible distortion of y given distortion of x = distort
    for all y in cone of angle theta with x.

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

    Maximum allowed distortion of x s.t. (distortion of y < distort) is
    guaranteed for all y in cone of angle theta with x.

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

    Assumes projection is onto first proj_dim dimensions

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
    return np.abs(np.sqrt(len(vec) * vec[0:proj_dim] @ vec[0:proj_dim] /
                          (proj_dim * vec @ vec)) - 1.)


# =============================================================================
# the whole calculation
# =============================================================================


def comparison(num_trials: int,
               theta: float,
               proj_dim: int,
               ambient_dim: int):
    """comparison of theory and experiment

    Comparison of theory and experiment
    Compute disortion of vectors between cell centres and vectors between edges
    of balls that enclose cells, to test assertion that:
        D_A(x) < E_C(epsilon,theta) ==> D_A(y) < epsilon for all y in C
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
    num_trials
        number of comparisons to find maximum distortion
    ambient_dim
        N, dimensionality of ambient space
    proj_dim
        M, dimensionality of projected space
    theta
        angle between centre and edge of chordal cone
    """
    x = make_x(ambient_dim)

    epsx = distortion(x, proj_dim)
    epsy = 0.

    for i in disp('trial', num_trials):
        dx = make_dx(x, theta)
        epsy = np.maximum(epsy, distortion(x + dx, proj_dim))

    gnt = guarantee(epsy, theta, proj_dim, ambient_dim)
    gnti = guarantee_inv(epsx, theta, proj_dim, ambient_dim)

    return epsx, gnt, epsy, gnti


def generate_data(num_trials: int,
                  ambient_dim: int,
                  thetas: Iterable[float],
                  proj_dims: Iterable[int],
                  num_reps: int):
    """generate all data for plots

    Generate all data for plots and legend
    Compute disortion of vectors between cell centres and vectors between edges
    of balls that enclose cells, to test assertion that:
        D_A(x) < E_C(epsilon,theta) ==> D_A(y) < epsilon for all y in C
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
    num_trials
        number of comparisons to find maximum distortion
    ambient_dim
        N, dimensionality of ambient space
    proj_dims
        M, list of dimensionalities of projected space
    thetas
        list of angles between centre and edge of chordal cone
    num_reps
        number of times to repeat each comparison
    """
    epsx = np.zeros((len(thetas), len(proj_dims), num_reps))
    gnt = np.zeros((len(thetas), len(proj_dims), num_reps))
    epsy = np.zeros((len(thetas), len(proj_dims), num_reps))
    gnti = np.zeros((len(thetas), len(proj_dims), num_reps))
    leg = []

    for i, theta in denum('theta', thetas):
        for j, M in denum('M', proj_dims):
            for r in disp('rep', num_reps):
                (epsx[i, j, r],
                 gnt[i, j, r],
                 epsy[i, j, r],
                 gnti[i, j, r]) = comparison(num_trials, theta, M, ambient_dim)
            leg.append(leg_text(i, j, thetas, proj_dims))
        # extra element at end of each row: label with value of theta
        leg.append(leg_text(i, len(proj_dims), thetas, proj_dims))

    return epsx, gnt, epsy, gnti, leg


# =============================================================================
# plotting
# =============================================================================


def leg_text(i: int,
             j: int,
             thetas: Iterable[float],
             proj_dims: Iterable[int]) -> str:
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
    num_trials
        number of comparisons to find maximum distortion
    ambient_dim
        N, dimensionality of ambient space
    thetas
        list of angles between centre and edge of chordal cone
    proj_dims
        M, list of dimensionalities of projected space
    num_reps
        number of times to repeat each comparison
    """
    # choose parameters
    np.random.seed(0)
    # number of samples of edge of cone
    num_trials = 2000000
    # number of times to repeat each comparison
    num_reps = 5
    # dimensionality of ambient space
    ambient_dim = 1000
    # dimensionality of projection
    proj_dims = [50, 75, 100]
    # angle between cone centre and edge
    thetas = [0.001, 0.002, 0.003, 0.004]

    return num_trials, ambient_dim, thetas, proj_dims, num_reps


def quick_options():
    """
    Demo options for generating test data

    Returns
    =======
    num_trials
        number of comparisons to find maximum distortion
    ambient_dim
        N, dimensionality of ambient space
    thetas
        list of angles between centre and edge of chordal cone
    proj_dims
        M, list of dimensionalities of projected space
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
    # angle between cone centre and edge
    thetas = [0.001, 0.002, 0.003]

    return num_trials, ambient_dim, thetas, proj_dims, num_reps


# =============================================================================
# running code
# =============================================================================


def make_and_save(filename: str,
                  num_trials: int,
                  ambient_dim: int,
                  thetas: Iterable[float],
                  proj_dims: Iterable[int],
                  num_reps: int):
    """
    Generate data and save in .npz file

    Parameters
    ==========
    filename
        name of .npz file, w/o extension, for data
    num_trials
        number of comparisons to find maximum distortion
    ambient_dim
        N, dimensionality of ambient space
    thetas
        list of angles between centre and edge of chordal cone
    proj_dims
        M, list of dimensionalities of projected space
    num_reps
        number of times to repeat each comparison
    """
    epsx, gnt, epsy, gnti, leg = generate_data(num_trials, ambient_dim, thetas,
                                               proj_dims, num_reps)
    np.savez_compressed(filename + '.npz', epsx=epsx, gnt=gnt, epsy=epsy,
                        gnti=gnti, leg=leg)


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
