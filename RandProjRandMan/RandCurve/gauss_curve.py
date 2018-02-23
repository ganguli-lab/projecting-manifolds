# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:46:16 2016

@author: Subhy

Compute distance, angle between tangent vectors and curvature as function of
position on a Gaussian random curve in a high dimensional space

Functions
=========
numeric_distance
    numeric distance between points on curve
numeric_cosines
    numeric angle between tangents to curve
numeric_curv
    numeric curvature of curve
get_all_numeric
    calculate all numeric quantities
make_fig_ax
    Make figure and axes objects
plot_theory_all
    Plot theory + 1 numeric graph for all of distance, angle, curvature
plot_num_all
    Plot numeric graph for all of distance, angle, curvature
save_figs_all
    save figures as pdfs
default_options_data
    default options for numerics
default_options_plot
    dicts and tuples of default options for plots
make_and_plot
    generate data and plot figures
"""
import numpy as np
from typing import Sequence
from . import gauss_curve_theory as gct
from . import gauss_curve_plot as gcp
from ..iter_tricks import dcount
# import matplotlib.pyplot as plt


# =============================================================================
# generate curve
# =============================================================================


def gauss_sqrt_cov_ft(k: np.ndarray, width: float=1.0) -> np.ndarray:
    """sqrt of FFT of Gaussian covariance matrix

    Square root of Fourier transform of a covariance matrix that is a Gaussian
    function of difference in position

    Returns
    -------
    cov(k)
        sqrt(sqrt(2pi) width * exp(-1/2 width**2 k**2))

    Parameters
    ----------
    k
        vector of spatial frequencies
    width
        std dev of gaussian covariance. Default=1.0
    """
    # length of grid
    num_pt = k.size
    # check if k came from np.fft.rfftfreq instead of np.fft.fftfreq
    if k.ravel()[-1] > 0:
        num_pt = 2. * (k.size - 1.)
    dk = k.ravel()[1]
    cov_ft = (dk / np.sqrt(2 * np.pi)) * width * np.exp(-0.5 * width**2 * k**2)
    return num_pt * np.sqrt(cov_ft)


def random_embed_ft(num_dim: int,
                    k: np.ndarray,
                    width: float=1.0) -> np.ndarray:
    """generate FFT of ramndom Gaussian curve

    Generate Fourier transform of ramndom Gaussian curve with a covariance
    matrix that is a Gaussian function of difference in position

    Returns
    -------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[t, i] = phi_i(k[t])

    Parameters
    ----------
    k
        vector of spatial frequencies
    num_dim
        dimensionality ofambient space
    width
        std dev of gaussian covariance
    """
    emb_ft_r = np.random.randn(k.size, num_dim) * gauss_sqrt_cov_ft(k, width)
    emb_ft_i = np.random.randn(k.size, num_dim) * gauss_sqrt_cov_ft(k, width)
    return (emb_ft_r + 1j * emb_ft_i) / np.sqrt(2 * num_dim)


# =============================================================================
# calculate intermediaries
# =============================================================================


def spatial_freq(intrinsic_range: float,
                 intrinsic_num: int,
                 expand: int=2) -> np.ndarray:
    """vector of spatial frequencies

    Vector of spatial frequencies

    Returns
    -------
    k[t]
        spatial frequencies. Appropriate singleton dimension added to
        broadcast with `embed_ft`

    Parameters
    ----------
    intrinsic_range
        range of intrinsic coord: [-intrinsic_range, intrinsic_range]
    intrinsic_num
        number of sampling points on curve
    expand
        factor to increase size by, to subsample later
    """
    intrinsic_res = 2. * intrinsic_range / intrinsic_num
    return 2 * np.pi * np.fft.rfftfreq(expand * intrinsic_num,
                                       intrinsic_res)[:, None]


def embed(embed_ft: np.ndarray) -> np.ndarray:
    """
    Calculate embedding functions

    Returns
    -------
    emb
        emb[t,i] = phi_i(x[t])

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[t,i] = phi_i(k[t])
    k
        vector of spatial frequencies
    """
    return np.fft.irfft(embed_ft, axis=0)


def embed_grad(embed_ft: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Calculate gradient of embedding functions

    Returns
    -------
    grad
        grad[t,i] = phi'_i(x[t])

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[t,i] = phi_i(k[t])
    k
        vector of spatial frequencies
    """
    return np.fft.irfft(embed_ft * (1j * k), axis=0)


def embed_hess(embed_ft: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Calculate hessian of embedding functions

    Returns
    -------
    hess
        hess[t,i] = phi''_i(x[t])

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[t,i] = phi_i(k[t])
    k
        vector of spatial frequencies
    """
    return np.fft.irfft(-embed_ft * k**2, axis=0)


def vielbein(grad: np.ndarray) -> np.ndarray:
    """
    Normalised tangent vector, push-forward of vielbein.

    Returns
    -------
    vbein
        normalised tangent vectors,
        vbein[t,i] = e^i(x[t]).

    Parameters
    ----------
    grad
        grad[t,i] = phi'_i(x[t])
    """
    u1 = grad / np.linalg.norm(grad, axis=-1, keepdims=True)
    return u1


# =============================================================================
# calculate distances, angles and curvature
# =============================================================================


def numeric_distance(embed_ft: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate Euclidean distance from central point on curve as a fuction of
    position on curve.

    Returns
    -------
    d
        chord length.
        ||phi(x[t]) - phi(x[mid])||
    ndx
        chord direction.
        ndx[t,i] = (phi_i(x[t]) - phi_i(x[mid])) / d

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[t,i] = phi_i(k[t])
    k
        vector of spatial frequencies
    """
    pos = embed(embed_ft)
    # chords
    dx = pos - pos[pos.shape[0]//2, :]
    # chord length
    d = np.linalg.norm(dx, axis=-1)
    zero = d < 1e-7
    d[zero] = 1.
    ndx = np.where(zero[..., None], 0., dx / d[..., None])
    d[zero] = 0.
    return d, ndx


def numeric_angle(vbein: np.ndarray) -> np.ndarray:
    """
    Cosine of angle between tangent vectors

    Returns
    -------
    sa
        sa[t] = sin(angle between unit tangent vectors at x[mid] and x[t])

    Parameters
    ----------
    vbein
        normalised tangent vectors,
        vbein[t,i] = e^i(x[t]).
    """
    tangent_dots = vbein @ vbein[vbein.shape[0]//2, :]
    tangent_dots[tangent_dots > 1.] = 1.
    return np.sqrt(1. - tangent_dots**2)


def numeric_proj(ndx: np.ndarray,
                 vbein: np.ndarray,
                 ind_range: slice) -> (np.ndarray, np.ndarray):
    """
    Cosine of angle between tangent vectors and chords

    Returns
    -------
    ca
        ca[t] = max_s (cos angle between tangent vector at x[s] and chord
        between x[mid] and x[t]).
    ca_mid
        ca_mid[t] = cos angle between tangent vector at x[(t+mid)/2] and chord
        between x[mid] and x[t].

    Parameters
    ----------
    d
        ||phi(x[t]) - phi(x[mid])||
    ndx
        chord direction.
        ndx[t,i] = (phi_i(x[t]) - phi_i(x[mid])) / d
    vbein
        normalised tangent vectors,
        vbein[t,i] = e^i(x[t]).
    """
    ndx[ndx.shape[0]//2, :] = vbein[ndx.shape[0]//2, :]
    tangent_dots = np.abs(vbein @ ndx[..., None]).squeeze()
#    tang_dots_mid = np.abs(vbein[ndx.shape[0]//4:-ndx.shape[0]//4, None, :]
#                           @ ndx[::2, :, None]).squeeze()
    return tangent_dots[:, ind_range].max(axis=1)  # , tang_dots_mid


def numeric_curv(grad: np.ndarray, hess: np.ndarray) -> np.ndarray:
    """
    Extrinsic curvature

    Returns
    -------
    kappa
        curavature

    Parameters
    ----------
    grad
        grad[t,i] = d phi_i(x[t]) / dx
    hess
        hess[t,i] = d^2 phi_i(x[t]) / dx^2
    """
    speed = np.sum(grad**2, axis=-1)
    accel = np.sum(hess**2, axis=-1)
    dotprod = np.sum(grad * hess, axis=-1)
    return (speed * accel - dotprod**2) / speed**3


# =============================================================================
# the whole thing
# =============================================================================


def get_all_numeric(ambient_dim: int,
                    intrinsic_range: float,
                    intrinsic_num: int,
                    expand: int=2) -> (np.ndarray,
                                       np.ndarray,
                                       Sequence[np.ndarray],
                                       np.ndarray):
    """calculate everything

    Calculate everything

    Returns
    -------
    nud
        numeric distances
    nua
        numeric cosines (tangent-tangent)
    nup
        numeric cosines (chord-tangent), tuple (max, mid)
    nuc
        numeric curvature

    Parameters
    ----------
    ambient_dim
        N, dimensionality of ambient space
    intrinsic_range
        range of intrinsic coord: [-intrinsic_range, intrinsic_range]
    intrinsic_num
        number of sampling points on curve
    expand
        factor to increase size by, to subsample later, must be even
    """
#    intrinsic_res = 4 * intrinsic_range / intrinsic_num
    k = spatial_freq(intrinsic_range, intrinsic_num, expand)

    embed_ft = random_embed_ft(ambient_dim, k)
    tangent_vec = embed_grad(embed_ft, k)
    hess = embed_hess(embed_ft, k)

    einbein = vielbein(tangent_vec)

    int_begin = (expand - 1) * intrinsic_num // 2
    int_end = intrinsic_num + int_begin
#    int_begm = (expand - 1) * intrinsic_num // 4
#    int_endm = intrinsic_num // 2 + int_begm

    num_dist, ndx = numeric_distance(embed_ft)
    num_ang = numeric_angle(einbein)
    num_proj = numeric_proj(ndx, einbein, slice(int_begin, int_end))
    num_curv = numeric_curv(tangent_vec, hess)

    nud = num_dist[int_begin:int_end]
    nua = num_ang[int_begin:int_end]
    nup = num_proj[int_begin:int_end]
    nuc = num_curv[int_begin:int_end]

    return nud, nua, nup, nuc


# =============================================================================
# options
# =============================================================================


def default_options_data() -> (int, int, float, int):
    """
    Default options for generating data

    Returns
    -------
    num_trials
        number of comparisons to find maximum distortion
    ambient_dim
        N, dimensionality of ambient space
    intrinsic_range
        range of intrinsic coord: [-intrinsic_range, intrinsic_range]
    intrinsic_num
        number of sampling points on curve
    """
    # choose parameters
    np.random.seed(0)
    ambient_dim = 1000     # dimensionality of ambient space
    intrinsic_range = 5.0  # x-coordinate lies between +/- this
    # width defaults to 1.0
    intrinsic_num = 1024   # number of points to sample
    num_trials = 5         # number of additional numerical comparisons

    return num_trials, ambient_dim, intrinsic_range, intrinsic_num


# =============================================================================
# running code
# =============================================================================


def make_and_plot(num_trials: int,
                  ambient_dim: int,
                  intrinsic_range: float,
                  intrinsic_num: int,
                  xlabs: gcp.Labels,
                  ylabs: gcp.Labels,
                  txtopts: gcp.Options,
                  legopts: gcp.Options,
                  leglocs: gcp.Labels) -> Sequence[gcp.Figure]:
    """
    Generate data and plot

    Parameters
    ----------
    num_trials
        number of comparisons to find maximum distortion
    ambient_dim
        N, dimensionality of ambient space
    intrinsic_range
        range of intrinsic coord: [-intrinsic_range, intrinsic_range]
    intrinsic_num
        number of sampling points on curve
    x/ylabs
        list of string for x/y axis labels
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    leglocs
        list of locations of legend

    Returns
    -------
    figs
        list of figure objects with plots
    """
    figs, axs = gcp.make_fig_ax(4)

    thr = gct.get_all_analytic(ambient_dim,
                               intrinsic_range,
                               intrinsic_num)
    num = get_all_numeric(ambient_dim,
                          intrinsic_range,
                          intrinsic_num)

    gcp.plot_theory_all(axs, thr[0], thr[1:], num,
                        xlabs, ylabs, leglocs, txtopts, legopts)

    for i in dcount('trial', num_trials):
        num = get_all_numeric(ambient_dim, intrinsic_range, intrinsic_num)
        gcp.plot_num_all(axs, thr[0], num)

    axs[3].set_ylim(bottom=0.0)

    return figs


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    """If file is run"""
    print('Run from outside package.')
