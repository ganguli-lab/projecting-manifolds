# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:18:20 2017

@author: Subhy

Analytically compute distance, principal angles between tangent spaces and
curvature as a function of position on a Gaussian random manifold in a high
dimensional space

Functions
=========
analytic_distance
    analytic distance between points on manifold
analytic_sines
    analytic angle between tangents to manifold
analytic_proj
    analytic angle between chords and tangents to manifold
analytic_curv
    analytic curvature of manifold
get_all_analytic
    calculate all analytic quantities as function of (x, y)
get_all_analytic_line
    calculate all analytic quantities as function of rho
"""
import numpy as np
from typing import Sequence, Tuple

# =============================================================================
# calculate distances, angles and curvature
# =============================================================================


def geo_dist_sq(dcoords: Sequence[np.ndarray],
                width: Sequence[float]) -> np.ndarray:
    """
    Geodesic distance matrix that is a Gaussian function of difference in posn

    Returns
    -------
    dist_mat
        matrix sum_a (dx_a^2 / width_a^2)

    Parameters
    ----------
    dcoords
        sequence of vectors of position differences
    width
        sequence of std devs of gaussian covariance along each intrinsic axis
    """
    rho = 0.
    for dx, w in zip(dcoords, width):
        rho = rho + dx**2 / w**2
    return rho


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
    is Gaussian.

    Returns
    -------
    (sin(theta_max), sin(theta_min))
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
    => cos(angle) = |1 - rho| exp(- rho / 2), exp(- rho / 2).
    All angles are either theta_max or theta_min
    """
    cossqmin = np.exp(-rho)
    cosqother = (1. - rho)**2 * cossqmin
    cosqmax = 1. * cosqother
    cosqmax[rho > 2.] = 1. * cossqmin[rho > 2.]
    cossqmin[rho > 2.] = 1. * cosqother[rho > 2.]
    return np.stack((np.sqrt(1.-cosqmax), np.sqrt(1.-cossqmin)), axis=-1)


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


def analytic_curv(K: int, siz: np.ndarray) -> np.ndarray:
    """
    Analytic solutions for extrinsic curvature when covariance is Gaussian

    Returns
    -------
    k
        curvatures,
        k1 = k2 =... = k

    Parameters
    ----------
    K
        intrinsic dimensionality of manifold
    siz
        shape of x,y,/// grid

    Notes
    -----
    If covariance of embedding coords is C_ij(x-x'),
    x = intrinsic coord of curve
    ij = ambient space indices = 1,...,N
    => curvature = 4 * C_ii(0)

    When C_ij(x) = delta_ij / N * exp(- x^2 / 2 width^2)
    => curvature = K+2
    """
    return (len(siz) + 2) * np.ones(siz)


# =============================================================================
# the whole thing
# =============================================================================


def get_all_analytic(ambient_dim: int,
                     intrinsic_range: Sequence[float],
                     intrinsic_num: Sequence[int],
                     width: Sequence[float]) -> (Tuple[np.ndarray, ...],
                                                 np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray,
                                                 np.ndarray):
    """
    Calculate everything

    Returns
    -------
    x, rho
        coordinates
    thd
        theoretical distances
    ths
        theoretical sines
        (sin theta_max, sin_theta_min)
    thp
        theoretical projection angles
    thc
        theoretical curvature


    Parameters
    ----------
    ambient_dim
        N, dimensionality of ambient space
    intrinsic_range
        tuple of ranges of intrinsic coords [-intrinsic_range, intrinsic_range]
    intrinsic_num
        tuple of numbers of sampling points on manifold
    width
        tuple of std devs of gaussian covariance along each intrinsic axis
    """
    x = [np.linspace(-irange, irange, num=inum, endpoint=False) for
         irange, inum in zip(intrinsic_range, intrinsic_num)]
    rho = geo_dist_sq(np.ix_(*tuple(x)), width)

    thd = analytic_distance(rho)
    tha = analytic_sines(rho)
    thp = analytic_proj(rho)
    thc = analytic_curv(rho.ndim, rho.shape)

    return x, rho, thd, tha, thp, thc


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

    ro = np.logspace(np.log10(rho.min() + 0.01), np.log10(rho.max()),
                     num=numpts)
    thd = analytic_distance(ro)
    tha = analytic_sines(ro)
    thp = analytic_proj(ro)
    thc = analytic_curv(rho.ndim, ro.shape)

    return ro, thd, tha, thp, thc
