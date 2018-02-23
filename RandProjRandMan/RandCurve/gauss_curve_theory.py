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
analytic_proj
    analytic angle between chords and tangents to curve
analytic_curv
    analytic curvature of curve
get_all_analytic
    calculate all analytic quantities
"""

import numpy as np
from typing import Sequence


def gauss_cov(x: np.ndarray, width: float=1.0) -> np.ndarray:
    """
    Covariance matrix that is a Gaussian function of difference in position

    Returns
    -------
    cov
        exp(-1/2 * dpos^2 / width^2)

    Parameters
    ----------
    x
        array of position differences
    width
        std dev of gaussian covariance
    """
    return np.exp(-0.5 * x**2 / width**2)


# =============================================================================
# calculate distances, angles and curvature
# =============================================================================


def analytic_distance(x: np.ndarray, width: float=1.0) -> np.ndarray:
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


def analytic_angle(x: np.ndarray, width: float=1.0) -> np.ndarray:
    """
    Analytic solutions for tangent vector angle when covariance is Gaussian

    Returns
    -------
    sin(angle)
        array of sines

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
    return np.sqrt(1. - ((1 - x**2/width**2) * gauss_cov(x, width))**2)


def analytic_proj(x: np.ndarray, width: float=1.0) -> np.ndarray:
    """
    Analytic solutions for best chord - tangent vector projection when
    covariance is Gaussian

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
    mid_maximum = np.sqrt(rho4 / np.sinh(rho4))
    other_maxima = np.ones_like(rho4) / np.sqrt(2 * np.e)
    return np.maximum(mid_maximum, other_maxima)


def analytic_curv(x: np.ndarray) -> np.ndarray:
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


def get_all_analytic(ambient_dim: int,
                     intrinsic_range: Sequence[float],
                     intrinsic_num: Sequence[int],
                     expand: int=2) -> (np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray):
    """calculate everything

    Calculate everything

    Returns
    -------
    xo
        x-coordinate
    thd
        theoretical distances
    tha
        theoretical sines (tangent-tangent)
    thp
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
    theory_ang = analytic_angle(x)
    theory_proj = analytic_proj(x)
    theory_curv = analytic_curv(x)

    int_begin = (expand - 1) * intrinsic_num // 2
    int_end = intrinsic_num + int_begin

    xo = x[int_begin:int_end]
    thd = theory_dist[int_begin:int_end]
    tha = theory_ang[int_begin:int_end]
    thp = theory_proj[int_begin:int_end]
    thc = theory_curv[int_begin:int_end]

    return xo, thd, tha, thp, thc


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    """If file is run"""
    print('Run from outside package.')
