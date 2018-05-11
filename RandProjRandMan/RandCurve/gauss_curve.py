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
from . import gauss_mfld as gm
from . import gauss_curve_plot as gcp
from ..iter_tricks import dcount


# =============================================================================
# the whole thing
# =============================================================================


def get_all_numeric(ambient_dim: int,
                    intrinsic_range: float,
                    intrinsic_num: int,
                    width: float=1.,
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
    width
        std dev of gaussian covariance. Default=1.0
    expand
        factor to increase size by, to subsample later, must be even
    """
    return gm.get_all_numeric(ambient_dim, (intrinsic_range,),
                              (intrinsic_num,), (width,), expand)
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


def get_all_analytic(ambient_dim: int,
                     intrinsic_range: float,
                     intrinsic_num: int,
                     width: float=1.) -> (np.ndarray, np.ndarray, np.ndarray,
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
    width
        std dev of gaussian covariance. Default=1.0
    """

    x, rho, thd, tha, thp, thc = gm.gmt.get_all_analytic(ambient_dim,
                                                         (intrinsic_range,),
                                                         (intrinsic_num,),
                                                         (width,))
    tha1 = np.where(rho < 2., tha[:, 0], tha[:, 1])
    return x[0], thd, tha1, thp, thc


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

    thr = get_all_analytic(ambient_dim,
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
