# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:46:16 2016

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
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# generate curve
# =============================================================================


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


def numeric_distance(embed_ft):  # Euclidean distance from centre
    """
    Calculate Euclidean distance from central point on curve as a fuction of
    position on curve.

    Returns
    -------
    d
        ||phi(x[t]) - phi(x[mid])||

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
    return np.linalg.norm(dpos, axis=-1)


def numeric_cosines(tangent_vecs):  # cosine of angle between tangent vectors
    """
    Cosine of angle between tangent vectors

    Returns
    -------
    ca
        ca[s,t] = dot product of unit tangent vectors at x[s] and x[t]

    Parameters
    ----------
    tangent_vecs
        normalised tangent vectors,
        vbein[t,i] = e^i(x[t]).
    """
    tangent_dots = tangent_vecs @ tangent_vecs[tangent_vecs.shape[0] // 2, :]
#    tangent_norms = np.linalg.norm(tangent_vecs, axis=-1)
#    tangent_norms *= tangent_norms[tangent_norms.shape[0] // 2]
    return tangent_dots  # / tangent_norms


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
        theoretical cosines
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
    theory_curv = analytic_curv(x)

    int_begin = (expand - 1) * intrinsic_num // 2
    int_end = intrinsic_num + int_begin

    xo = x[int_begin:int_end]
    thd = theory_dist[int_begin:int_end]
    tha = theory_cos[int_begin:int_end]
    thc = theory_curv[int_begin:int_end]

    return xo, thd, tha, thc


def get_all_numeric(ambient_dim, intrinsic_range, intrinsic_num, expand=2):
    """calculate everything

    Calculate everything

    Returns
    -------
    nud
        numeric distances
    nua
        numeric cosines
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

    num_dist = numeric_distance(embed_ft)
    num_cos = numeric_cosines(einbein)
    num_curv = numeric_curv(tangent_vec, hess)

    int_begin = (expand - 1) * intrinsic_num // 2
    int_end = intrinsic_num + int_begin

    nud = num_dist[int_begin:int_end]
    nua = num_cos[int_begin:int_end]
    nuc = num_curv[int_begin:int_end]

    return nud, nua, nuc


# =============================================================================
# plotting
# =============================================================================


def make_fig_ax():  # make figure and axes objects
    """
    Make figure and axes objects

    Returns
    -------
    figs
        list of figure objects
    axs
        list of axes objects
    """
    figs = []
    axs = []
    for gsi in range(3):
        figs.append(plt.figure())
        axs.append(figs[-1].add_subplot(1, 1, 1))
    return figs, axs


def plot_theory(ax, x, thry, numl, xlab, ylab, legloc, textopts, lgtxtopt):
    """plot theory + simulation

    Plot theory and simulation

    Parameters
    ----------
    ax
        axes object to plot on
    x
        intrinsic coords
    thry
        theoretical result
    numl
        numerical result
    x/ylab
        string for x/y axis label
    legloc
        location of legend
    textopts
        text style options for axis labels
    lgtxtopt
        style options for legend
    """

    if 'prop' in lgtxtopt and 'size' in lgtxtopt['prop']:
        ax.tick_params(axis='both', which='major',
                       labelsize=lgtxtopt['prop']['size'])

    lin = ax.plot(x, thry, 'r-', x, numl, 'g-')
    lin[0].set_linewidth(2.0)
    lin[0].zorder = 20

    ax.set_xlabel(xlab, **textopts)
    ax.set_ylabel(ylab, **textopts)
    ax.legend(lin, ['Theory', 'Simulation'], loc=legloc, **lgtxtopt)


def plot_num(ax, x, numl):  # plot simulation
    """
    Plot simulation

    Parameters
    ----------
    ax
        axes object to plot on
    x
        intrinsic coords
    numl
        numerical result

    Notes
    -----
    Assumes plot_theory already used, so axis labels & legend already done
    """
    ax.plot(x, numl, 'g-')


def plot_theory_all(axs, x, thrys, numls, xlabls, ylabls, leglocs, textopts,
                    lgtxtopt):  # plot theory + simulation
    """
    Plot theory and simulation

    Parameters
    ----------
    axs
        list of axes objects to plot on for [distance, cosine, curvature]
    x
        intrinsic coords
    thrys
        list of theoretical results
    numls
        list of numerical results
    x/ylabs
        list of string for x/y axis labels
    leglocs
        list of locations of legend
    textopts
        text style options for axis labels
    lgtxtopt
        style options for legend
    """
    for ax, thr, numl, xlabl, ylabl, legloc in zip(axs, thrys, numls, xlabls,
                                                   ylabls, leglocs):
        plot_theory(ax, x, thr, numl, xlabl, ylabl, legloc, textopts, lgtxtopt)
        ax.grid(b=True)


def plot_num_all(axs, x, numls):  # plot simulation
    """
    Plot simulation

    Parameters
    ----------
    axs
        list of axes objects to plot on for [distance, cosine, curvature]
    x
        intrinsic coords
    numls
        list of numerical results

    Notes
    -----
    Assumes plot_theory already used, so axis labels & legend already done
    """
    for ax,  numl, in zip(axs, numls):
        plot_num(ax, x, numl)


def save_figs_all(figs, fignames, figpath):  # Save all figs
    """
    Save all figs to .pdf files

    Parameters
    ----------
    figs
        list of figure objects for [distance, cosine, curvature]
    fignames
        list of .pdf file names, w/o extensions or paths
    figpath
        path to folder for .pdf files, ending with '/'
    """
    for fig, fname in zip(figs, fignames):
        fig.tight_layout()
        fig.savefig(figpath + fname + '.pdf', bbox_inches='tight')


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


def default_options_plot():
    """
    Default options for plotting data

    Returns
    -------
    x/ylabs
        list of string for x/y axis labels
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    leglocs
        list of locations of legend
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode'] = True
    mpl.rcParams['font.family'] = r'serif'

    txtopts = {'size': 26, 'family': 'serif'}    #
    legopts = {'prop': {'size': 16, 'family': 'serif'},
               'frameon': False, 'framealpha': 0.}   #

    xlab = r'$(\sigma-\sigma^\prime)/\lambda$'
    xlab2 = r'$\sigma/\lambda$'
    dlab = r'$\Vert\phi(\sigma)-\phi(\sigma^\prime)\Vert/\ell$'
    alab = r'$\cos\theta$'
    clab = r'$\mathcal{K}\ell^2$'
#    xlab = r'Position difference, $(\sigma-\sigma^\prime)/\lambda$'
#    xlab2 = r'Position, $\sigma/\lambda$'
#    dlab = (r'Euclidean distance, ' +
#            r'$\Vert\phi(\sigma)-\phi(\sigma^\prime)\Vert/\ell$')
#    alab = r'Cosine tangent angle, $\cos\theta$'
#    clab = r'Curvature, $\mathcal{K}\ell^2$'

    xlabs = [xlab, xlab, xlab2]
    ylabs = [dlab, alab, clab]
    leglocs = ['lower right', 'upper right', 'lower right']

    return xlabs, ylabs, txtopts, legopts, leglocs


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
    figs, axs = make_fig_ax()

    x, thr_dis, thr_cos, thr_cur = get_all_analytic(ambient_dim,
                                                    intrinsic_range,
                                                    intrinsic_num)
    num_dis, num_cos, num_cur = get_all_numeric(ambient_dim,
                                                intrinsic_range,
                                                intrinsic_num)

    plot_theory_all(axs, x, [thr_dis, thr_cos, thr_cur],
                    [num_dis, num_cos, num_cur],
                    xlabs, ylabs, leglocs, txtopts, legopts)

    for i in range(num_trials):
        num_dis, num_cos, num_cur = get_all_numeric(ambient_dim,
                                                    intrinsic_range,
                                                    intrinsic_num)
        plot_num_all(axs, x, [num_dis, num_cos, num_cur])

    axs[2].set_ylim(bottom=0.0)

    return figs


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    """If file is run"""
    print('Run from outside package.')
