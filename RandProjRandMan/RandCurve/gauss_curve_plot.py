# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:26:08 2017

@author: Subhy

Compute distance, angle between tangent vectors and curvature as function of
position on a Gaussian random curve in a high dimensional space

Functions
=========
make_fig_ax
    Make figure and axes objects
plot_theory_all
    Plot theory + 1 numeric graph for all of distance, angle, curvature
plot_num_all
    Plot numeric graph for all of distance, angle, curvature
save_figs_all
    save figures as pdfs
default_options_plot
    dicts and tuples of default options for plots
make_and_plot
    generate data and plot figures
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Mapping, Any, Union

Options = Mapping[str, Any]
Labels = Sequence[str]
Axes = mpl.axes.Axes
Figure = mpl.figure.Figure

# =============================================================================
# plotting
# =============================================================================


def make_fig_ax(num: int=4) -> (Sequence[Figure], Sequence[Axes]):
    """
    Make figure and axes objects

    Returns
    -------
    figs
        list of figure objects
    axs
        list of axes objects

    Parameters
    ----------
    num
        number of figures
    """
    figs = [plt.figure() for i in range(num)]
    axs = [fig.add_subplot(1, 1, 1) for fig in figs]
    return figs, axs


def plot_theory(ax: Axes,
                x: np.ndarray,
                thry: np.ndarray,
                numl: Union[np.ndarray, Sequence[np.ndarray]],
                xlab: str,
                ylab: str,
                legloc: str,
                textopts: Options,
                lgtxtopt: Options):
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

    if isinstance(numl, tuple):
        lin = ax.plot(x, thry, 'r-', x, numl[0], 'g-', x[::2], numl[1], 'b-')
        leg = ['Theory', 'Simulation', 'Sim mid']
    else:
        lin = ax.plot(x, thry, 'r-', x, numl, 'g-')
        leg = ['Theory', 'Simulation']

    lin[0].set_linewidth(2.0)
    lin[0].zorder = 20

    ax.set_xlabel(xlab, **textopts)
    ax.set_ylabel(ylab, **textopts)
    ax.legend(lin, leg, loc=legloc, **lgtxtopt)


def plot_num(ax: Axes,
             x: np.ndarray,
             numl: np.ndarray):  # plot simulation
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
    if isinstance(numl, tuple):
        ax.plot(x, numl[0], 'g-', x[::2], numl[1], 'b-')
    else:
        ax.plot(x, numl, 'g-')


def plot_theory_all(axs: Sequence[Axes],
                    x: np.ndarray,
                    thrys: Sequence[np.ndarray],
                    numls: Sequence[np.ndarray],
                    xlabls: Sequence[str],
                    ylabls: Sequence[str],
                    leglocs: Sequence[str],
                    textopts: Options,
                    lgtxtopt: Options):  # plot theory + simulation
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


def plot_num_all(axs: Sequence[Axes],
                 x: np.ndarray,
                 numls: Sequence[np.ndarray]):  # plot simulation
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


def save_figs_all(figs: Sequence[Figure],
                  fignames: Sequence[str],
                  figpath: str):  # Save all figs
    """
    Save all figs to ``.pdf`` files

    Parameters
    ----------
    figs
        list of figure objects for [distance, cosine, curvature]
    fignames
        list of ``.pdf`` file names, w/o extensions or paths
    figpath
        path to folder for ``.pdf`` files, ending with '/'
    """
    for fig, fname in zip(figs, fignames):
        fig.tight_layout()
        fig.savefig(figpath + fname + '.pdf', bbox_inches='tight')


# =============================================================================
# options
# =============================================================================


def default_options_plot() -> (Sequence[str], Sequence[str],
                               Options, Options,
                               Sequence[str]):
    """
    Default options for plotting data

    Returns
    -------
    x/ylabs
        list of string for x/y axis labels
    leglocs
        list of locations of legend
    txtopts
        text style options for axes labels
    legopts
        style options for legend
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
    alab = r'$\sin\theta_{\mathcal{T}}$'
    tlab = r'$\max|\cos\theta_{\mathcal{S}}|$'
    clab = r'$\mathcal{K}\ell^2$'
#    xlab = r'Position difference, $(\sigma-\sigma^\prime)/\lambda$'
#    xlab2 = r'Position, $\sigma/\lambda$'
#    dlab = (r'Euclidean distance, ' +
#            r'$\Vert\phi(\sigma)-\phi(\sigma^\prime)\Vert/\ell$')
#    alab = r'Cosine tangent angle, $\cos\theta$'
#    clab = r'Curvature, $\mathcal{K}\ell^2$'

    xlabs = [xlab, xlab, xlab, xlab2]
    ylabs = [dlab, alab, tlab, clab]
    leglocs = ['lower right', 'lower right', 'lower center', 'lower right']

    return xlabs, ylabs, leglocs, txtopts, legopts


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    """If file is run"""
    print('Run from outside package.')
