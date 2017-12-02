# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:18:20 2017

@author: Subhy

Analytically compute distance, principal angles between tangent spaces and
curvature as a function of position on a Gaussian random surface in a high
dimensional space

Functions
=========
analytic_distance
    analytic distance between points on surface
analytic_sines
    analytic angle between tangents to surface
analytic_proj
    analytic angle between chords and tangents to surface
analytic_curv
    analytic curvature of surface
get_all_analytic
    calculate all analytic quantities as function of (x, y)
get_all_analytic_line
    calculate all analytic quantities as function of rho
"""
import numpy as np
from typing import Sequence

# =============================================================================
# calculate distances, angles and curvature
# =============================================================================


def geo_dist_sq(dx: np.ndarray,
                dy: np.ndarray,
                width: Sequence[float]=(1.0, 1.0)) -> np.ndarray:
    """
    Geodesic distance matrix that is a Gaussian function of difference in posn

    Returns
    -------
    dist_mat
        matrix sum_a (dx_a^2 / width_a^2)

    Parameters
    ----------
    dx,dy
        vectors of position differences
    width
        tuple of std devs of gaussian covariance along each intrinsic axis
    """
    return dx[:, np.newaxis]**2 / width[0]**2 + dy**2 / width[1]**2


def analytic_distance(rho: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distance from central point on curve as a fuction of
    position on curve.

    Returns
    -------
    d
        d[s] = ||phi(x[s]) - phi(x[mid])||

    Parameters
    ----------
    rho
        rho = sum_a (dx_a^2 / width_a^2)
    """
    return np.sqrt(2 * (1 - np.exp(-rho / 2.)))


def analytic_sines(rho: np.ndarray) -> np.ndarray:
    """
    Analytic solutions for tangent space principal angle sines  when covariance
    is Gaussian

    Returns
    -------
    sin(theta_max), sin(theta_min)
        S[a][s,t] = tuple of sin theta_a[s,t]
    theta_a
        principal angle between tangent spaces at x[s], x[t] and at center

    Parameters
    ----------
    rho
        sum_a (x_a^2 / width_a^2)
    x_a
        position vectors

    Notes
    -----
    When C_ij(x) = delta_ij / N * exp(-rho / 2)
    => cos(angle) = |1 - rho| exp(- rho / 2), exp(- rho / 2)
    """
    cossqmin = np.exp(-rho)
    cosqother = (1. - rho)**2 * cossqmin
    cosqmax = 1. * cosqother
    cosqmax[rho > 2.] = 1. * cossqmin[rho > 2.]
    cossqmin[rho > 2.] = 1. * cosqother[rho > 2.]
    return np.sqrt(1. - cosqmax), np.sqrt(1. - cossqmin)
#    Returns S[s,t] = sum_a sin^2 theta_a[s,t]
#    return 2 - (2 - 2 * rho + rho**2) * gauss_cov(dx, dy, width)**2


def analytic_proj(rho: np.ndarray) -> np.ndarray:  # Analytic sum squared sines
    """
    Analytic solutions for tangent space principal angle sines  when covariance
    is Gaussian

    Returns
    -------
    cos(theta_max)
        C[a][s,t] = max_u,v {cos theta[s,t,u,v]}
    theta[s,t,u,v]
        angle between tangent spaces at x[u], y[v] and chord between x[s], y[t]
        and center

    Parameters
    ----------
    rho
        sum_a (x_a^2 / width_a^2)
    x_a
        position vectors

    Notes
    -----
    When C_ij(x) = delta_ij / N * exp(-rho / 2)
    => cos(angle) = |1 - rho| exp(- rho / 2), exp(- rho / 2)
    """
    rho4 = rho / 4.
    rho4[rho4 <= 1e-18] = 1e-18
    return np.sqrt(rho4 / np.sinh(rho4))


def analytic_curv(siz: np.ndarray) -> np.ndarray:
    """
    Analytic solutions for extrinsic curvature when covariance is Gaussian

    Returns
    -------
    k
        curvatures,
        k1 = k2 = k

    Parameters
    ----------
    siz
        shape of x,y grid

    Notes
    -----
    If covariance of embedding coords is C_ij(x-x'),
    x = intrinsic coord of curve
    ij = ambient space indices = 1,...,N
    => curvature = 4 * C_ii(0)

    When C_ij(x) = delta_ij / N * exp(- x^2 / 2 width^2)
    => curvature = 4
    """
    return 4 * np.ones(siz)


# =============================================================================
# the whole thing
# =============================================================================


def get_all_analytic(ambient_dim: int,
                     intrinsic_range: Sequence[float],
                     intrinsic_num: Sequence[int],
                     width: Sequence[float]=(1.0, 1.0),
                     expand: int=2) -> (np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate everything

    Returns
    -------
    x, y, rho
        coordinates
    thd
        theoretical distances
    ths
        theoretical sines
        (sin theta_max, sin_theta_min)
    thc
        theoretical curvature


    Parameters
    ----------
    ambient_dim
        N, dimensionality of ambient space
    intrinsic_range
        tuple of ranges of intrinsic coords [-intrinsic_range, intrinsic_range]
    intrinsic_num
        tuple of numbers of sampling points on surface
    width
        tuple of std devs of gaussian covariance along each intrinsic axis
    expand
        factor to increase size by, to subsample later
    """
    x = np.linspace(-expand * intrinsic_range[0], expand * intrinsic_range[0],
                    num=expand * intrinsic_num[0], endpoint=False)
    y = np.linspace(-expand * intrinsic_range[1], expand * intrinsic_range[1],
                    num=expand * intrinsic_num[1], endpoint=False)
    rho = geo_dist_sq(x, y, width)

    theory_dist = analytic_distance(rho)
    theory_sin_max, theory_sin_min = analytic_sines(rho)
    theory_pr = analytic_proj(rho)
    theory_curv = analytic_curv(rho.shape)

    int_begin = ((expand - 1) * intrinsic_num[0] // 2,
                 (expand - 1) * intrinsic_num[1] // 2)
    int_end = (intrinsic_num[0] + int_begin[0],
               intrinsic_num[1] + int_begin[1])
    region = (slice(int_begin[0], int_end[0]),
              slice(int_begin[1], int_end[1]))

    xo = x[region[0]]
    yo = y[region[1]]
    ro = rho[region]
    thd = theory_dist[region]
    tha = (theory_sin_max[region], theory_sin_min[region])
    thp = theory_pr[region]
    thc = theory_curv[region]

    return xo, yo, ro, thd, tha, thp, thc


def get_all_analytic_line(rho: np.ndarray,
                          numpts: int) -> (np.ndarray, np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray):
    """
    Calculate everything

    Returns
    -------
    rho_line
        rho.min() to rho.maxc(), linearly spaced
    theoretical distances
        ambient space distances as fcn of intrinsic coords
    theoretical sines
        (sin theta_max, sin_theta_min)
    theoretical curvature
        extrinsic curvature evals as fcn of intrinsic coords
    rho
        sum_a (dx_a^2 / width_a^2)
    numpts
        number of points to use
    """

    ro = np.logspace(np.log10(rho.min() + 0.01),
                     np.log10(rho.max()), num=numpts)
    thd = analytic_distance(ro)
    tha = analytic_sines(ro)
    thp = analytic_proj(ro)
    thc = analytic_curv(ro.shape)

    return ro, thd, tha, thp, thc
