# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:16:57 2017

@author: Subhy

Compute distance, angle between tangent vectors and curvature as function of
position on a Gaussian random curve in a high dimensional space

Functions
=========
analytic_distance
    analytic distance between points on curve
analytic_cosines
    analytic angle between tangents to curve
analytic_curv
    analytic curvature of curve
get_all_analytic
    calculate all analytic quantities
"""

import numpy as np


def gauss_cov(x, width=1.0):  # Gaussian covariance matrix
    """
    Covariance matrix that is a Gaussian function of difference in position

    Returns
    -------
    cov
        exp(-1/2 * dpos^2 / width^2)

    x
        array of position differences
    width
        std dev of gaussian covariance
    """
    return np.exp(-0.5 * x**2 / width**2)


# =============================================================================
# calculate distances, angles and curvature
# =============================================================================


def analytic_distance(x, width=1.0):  # Euclidean distance from centre
    """
    Calculate Euclidean distance from central point on curve as a fuction of
    position on curve.

    Returns
    -------
    d
        ||phi(x[i]) - phi(x[mid])||

    Parameters
    ----------
    x
        intrinsic coord of curve
    width
        std dev of gaussian covariance
    """
    return np.sqrt(2 * (1 - gauss_cov(x, width)))


def analytic_cosines(x, width=1.0):  # Analytic soln for cosine tangent angle
    """
    Analytic solutions for tangent vector cosine when covariance is Gaussian

    Returns
    -------
    cos(angle)
        array of cosines

    Parameters
    ----------
    x
        intrinsic coord of curve
    width
        std dev of gaussian covariance

    Notes
    -----
    If covariance of embedding coords is C_ij(x-x'),
    ij = ambient space indices = 1,...,N
    angle(x,x') = angle between tangent vectors at x,x'
    Let H(x) = - sum_i C''_ii(x)
    Than cos(angle) = H(x-x')/|H(0)|

    When C_ij(x) = delta_ij / N * exp(- x^2 / 2 width^2)
    => cos(angle) = (1 - x^2 / width^2) exp(- x^2 / 2 width^2)
    """
    return (1 - x**2 / width**2) * gauss_cov(x, width)


def analytic_tangcosines(x, width=1.0):  # Analytic soln for chord angle
    """
    Analytic solutions for tangent vector cosine when covariance is Gaussian

    Returns
    -------
    cos(angle)
        array of cosines

    Parameters
    ----------
    x
        intrinsic coord of curve
    width
        std dev of gaussian covariance

    Notes
    -----
    If covariance of embedding coords is C_ij(x-x'),
    ij = ambient space indices = 1,...,N
    angle(x,x') = angle between tangent vectors at x,x'
    Let H(x) = - sum_i C''_ii(x)
    Than cos(angle) = H(x-x')/|H(0)|

    When C_ij(x) = delta_ij / N * exp(- x^2 / 2 width^2)
    => cos(angle) = (1 - x^2 / width^2) exp(- x^2 / 2 width^2)
    """
    rho4 = x**2 / (2 * width)**2
    rho4[rho4 <= 1e-18] = 1e-18
    return np.sqrt(rho4 / np.sinh(rho4))


def analytic_curv(x):  # Analytic solution for extrinsic curvature
    """
    Analytic solutions for extrinsic curvature when covariance is Gaussian

    Returns
    -------
    kappa
        curvature

    Parameters
    ----------
    x
        intrinsic coord of curve

    Notes
    -----
    If covariance of embedding coords is C_ij(x-x'),
    ij = ambient space indices = 1,...,N
    => curvature = 3 * C_ii(0)

    When C_ij(x) = delta_ij / N * exp(- x^2 / 2 width^2)
    => curvature = 3
    """
    return 3 * np.ones(x.shape)


# =============================================================================
# the whole thing
# =============================================================================


def get_all_analytic(ambient_dim, intrinsic_range, intrinsic_num, expand=2):
    """calculate everything

    Calculate everything

    Returns
    -------
    xo
        x-coordinate
    thd
        theoretical distances
    tha
        theoretical cosines (tangent-tangent)
    tht
        theoretical cosines (chord-tangent)
    thc
        theoretical curvature

    Parameters
    ----------
    ambient_dim
        N, dimensionality of ambient space
    intrinsic_range
        range of intrinsic coord: [-intrinsic_range, intrinsic_range]
    intrinsic_num
        number of sampling points on curve
    expand
        factor to increase size by, to subsample later
    """
    x = np.linspace(-expand * intrinsic_range, expand * intrinsic_range,
                    num=expand * intrinsic_num, endpoint=False)

    theory_dist = analytic_distance(x)
    theory_cos = analytic_cosines(x)
    theory_tan = analytic_tangcosines(x)
    theory_curv = analytic_curv(x)

    int_begin = (expand - 1) * intrinsic_num // 2
    int_end = intrinsic_num + int_begin

    xo = x[int_begin:int_end]
    thd = theory_dist[int_begin:int_end]
    tha = theory_cos[int_begin:int_end]
    tht = theory_tan[int_begin:int_end]
    thc = theory_curv[int_begin:int_end]

    return xo, thd, tha, tht, thc


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    """If file is run"""
    print('Run from outside package.')
