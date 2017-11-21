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
from . import gauss_curve_theory as gct
from . import gauss_curve_plot as gcp

# =============================================================================
# generate curve
# =============================================================================


def gauss_sqrt_cov_ft(k, width=1.0):
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
    return (num_pt * np.sqrt((k.ravel()[1] / np.sqrt(2 * np.pi)) * width *
                             np.exp(-0.5 * width**2 * k**2)))


def random_embed_ft(num_dim, k, width=1.0):
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


def spatial_freq(intrinsic_range, intrinsic_num, expand=2):
    """vector of spatial frequencies

    Vector of spatial frequencies

    Returns
    -------
    k[t]
        spatial frequencies. Appropriate singleton dimension added to
        broadcast with embed_ft

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


def embed(embed_ft):  # calculate embedding functions
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


def embed_grad(embed_ft, k):  # calculate gradient of embedding functions
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


def embed_hess(embed_ft, k):  # calculate hessian of embedding functions
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


def vielbein(grad):  # orthonormal basis for tangent space
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


def numeric_distance(embed_ft):  # Euclidean distance from centre
    """
    Calculate Euclidean distance from central point on curve as a fuction of
    position on curve.

    Returns
    -------
    d
        ||phi(x[t]) - phi(x[mid])||
    dx
        dx[t,i] = phi_i(x[t]) - phi_i(x[mid])

    Parameters
    ----------
    embed_ft
        Fourier transform of embedding functions,
        embed_ft[t,i] = phi_i(k[t])
    k
        vector of spatial frequencies
    """
    pos = embed(embed_ft)
    dpos = pos - pos[pos.shape[0] // 2, :]
    return np.linalg.norm(dpos, axis=-1), dpos


def numeric_cosines(vbein):  # cosine of angle between tangent vectors
    """
    Cosine of angle between tangent vectors

    Returns
    -------
    ca
        ca[t] = dot product of unit tangent vectors at x[mid] and x[t]

    Parameters
    ----------
    vbein
        normalised tangent vectors,
        vbein[t,i] = e^i(x[t]).
    """
    tangent_dots = vbein @ vbein[vbein.shape[0] // 2, :]
    return tangent_dots


def numeric_tangcosines(dx, d, vbein):  # angle between tangent vectors
    """
    Cosine of angle between tangent vectors and chords

    Returns
    -------
    ca
        ca[t] = max_s (cos angle between tangent vector at x[s] and chord
        between x[mid] and x[t].)

    Parameters
    ----------
    dx
        dx[t,i] = phi_i(x[t]) - phi_i(x[mid])
    d
        ||phi(x[t]) - phi(x[mid])||
    vbein
        normalised tangent vectors,
        vbein[t,i] = e^i(x[t]).
    """
    ndx = np.where(d[..., None] > 1e-7, dx / d[..., None], vbein)
    tangent_dots = np.abs(vbein @ ndx[..., None])
    return tangent_dots.max(axis=1).ravel()


def numeric_curv(grad, hess):  # curvature of curve
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


def get_all_numeric(ambient_dim, intrinsic_range, intrinsic_num, expand=2):
    """calculate everything

    Calculate everything

    Returns
    -------
    nud
        numeric distances
    nua
        numeric cosines (tangent-tangent)
    nut
        numeric cosines (chord-tangent)
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
        factor to increase size by, to subsample later
    """
#    intrinsic_res = 4 * intrinsic_range / intrinsic_num
    k = spatial_freq(intrinsic_range, intrinsic_num, expand)

    embed_ft = random_embed_ft(ambient_dim, k)
    tangent_vec = embed_grad(embed_ft, k)
    hess = embed_hess(embed_ft, k)

    einbein = vielbein(tangent_vec)

    num_dist, dx = numeric_distance(embed_ft)
    num_cos = numeric_cosines(einbein)
    num_tan = numeric_tangcosines(dx, num_dist, einbein)
    num_curv = numeric_curv(tangent_vec, hess)

    int_begin = (expand - 1) * intrinsic_num // 2
    int_end = intrinsic_num + int_begin

    nud = num_dist[int_begin:int_end]
    nua = num_cos[int_begin:int_end]
    nut = num_tan[int_begin:int_end]
    nuc = num_curv[int_begin:int_end]

    return nud, nua, nut, nuc


# =============================================================================
# options
# =============================================================================


def default_options_data():
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


def make_and_plot(num_trials, ambient_dim, intrinsic_range, intrinsic_num,
                  xlabs, ylabs, txtopts, legopts, leglocs):
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

    for i in range(num_trials):
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
