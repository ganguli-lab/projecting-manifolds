# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:14:44 2016

@author: Subhy

Plot disortion of tangent space at cell centre and tangent spaces at edge of
a Grassmannian region that encloses the image of cell under the Gauss map,
to test assertion that:

.. math::
    D_A(U) < E_T(\\epsilon,\\theta_T) \\implies D_A(U') < \\epsilon
                                             \\;\\forall U' \\in T
| where T = tangential cone,
| :math:`\\theta_T` > principal angles between U and U',
| U = central subspace
"""

from typing import Sequence, Mapping, Any
import numpy as np
from numpy import ndarray as array
import matplotlib as mpl
import matplotlib.pyplot as plt

Styles = Sequence[Mapping[str, str]]
Options = Mapping[str, Any]
Labels = Sequence[str]
Axes = mpl.axes.Axes
Figure = mpl.figure.Figure


# =============================================================================
# plotting
# =============================================================================


def plot_equality(ax: Axes):  # plot x=y line
    """
    Plots line showing where x=y on axes `ax`
    """
    xl = ax.get_xlim()
    yl = ax.get_ylim()

    lim = np.array([np.maximum(xl[0], yl[0]), np.minimum(xl[1], yl[1])])

    ax.plot(lim, lim, 'k-')
    ax.set_xlim(xl)
    ax.set_ylim(yl)


def plot_scatter(ax: Axes,
                 eps: array,
                 gnt: array,
                 labels: Labels,
                 leg: Labels,
                 pst: Styles,
                 psk: Styles,
                 psm: Styles,
                 txtopts: Options,
                 legopts: Options):
    """
    Plot all data and legend

    Parameters
    ----------
    ax
        axes object for plotting on
    epsx
        distortions
    gnt
        guarantees
    labels
        list of axes labels [x, y]
    leg
        legend text associated with corresponding datum, or ``None``
    pst
        list of plot styles associated with `thetas`
    psm
        list of plot styles associated with `proj_dims`
    psk
        list of plot styles associated with `sub_dims`
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    """

    if 'prop' in legopts and 'size' in legopts['prop']:
        ax.tick_params(axis='both', which='major',
                       labelsize=legopts['prop']['size'])

    lhs = []

    for t, pt in enumerate(pst[:eps.shape[0]]):
        for k, pk in enumerate(psk[:eps.shape[1]]):
            for m, pm in enumerate(psm[:eps.shape[2]]):
                lhs.append(ax.plot(eps[t, k, m], gnt[t, k, m],
                                   **pt, **pm, **pk, linestyle='none')[0])
            # extra element at end of each row: label with value of M
            lhs.append(ax.plot(eps[t, k, 0], gnt[t, k, 0],
                               **pt, **pm, **psk[0], linestyle='none')[0])
        # extra element at end of each column: label with value of theta
        lhs.append(ax.plot(eps[t, 0, 0], gnt[t, 0, 0],
                           **pt, **psm[0], **psk[0], linestyle='none')[0])

    ax.set_xlabel(labels[0], **txtopts)
    ax.set_ylabel(labels[1], **txtopts)

    if leg is not None:
        mstep = gnt.shape[1]
        kstep = gnt.shape[2]
        lhs = (lhs[0:kstep] + lhs[kstep:(kstep + 1) * mstep:kstep + 1]
               + lhs[(kstep + 1) * mstep::(kstep + 1) * mstep + 1])
        leg = (leg[0:kstep] + leg[kstep:(kstep + 1) * mstep:kstep + 1]
               + leg[(kstep + 1) * mstep::(kstep + 1) * mstep + 1])
        ax.legend(lhs, leg, numpoints=1, **legopts)

    plot_equality(ax)


def plot_data(ax: Axes,
              eps: array,
              gnt: array,
              leg: Labels,
              pst: Styles,
              psk: Styles,
              psm: Styles,
              txtopts: Options,
              legopts: Options):
    """plot all data

    Plot all data and legend

    Parameters
    ----------
    ax
        axes object for plotting on
    eps
        distortion of central subspace U
    gnt
        guarantee(maximum distortion of U' for U' in tangential cone
    leg
        legend text associated with corresponding datum
    pst
        list of plot styles associated with `thetas`
    psm
        list of plot styles associated with `proj_dims`
    psk
        list of marker fill styles associated with `sub_dims`
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    """

    labels = [r'$\mathcal{D}_{\mathbf{A}}(\mathcal{U})$',
              r'$\mathcal{E}_\mathcal{T}(\max\mathcal{D}_{\mathbf{A}}'
              + r'(\mathcal{U}^\prime),\theta_{\mathcal{T}})$']
    plot_scatter(ax, eps, gnt, labels, leg, pst, psm, psk, txtopts, legopts)
    ax.yaxis.set_label_coords(-0.1, 0.5)


def plot_data_inv(ax: Axes,
                  epsb: array,
                  gnti: array,
                  leg: Labels,
                  pst: Styles,
                  psk: Styles,
                  psm: Styles,
                  txtopts: Options,
                  legopts: Options):
    """plot all data

    Plot all data and legend

    Parameters
    ----------
    ax
        axes object for plotting on
    epsb
        maximum distortion of U' for U' in tangential cone
    gnti
        guarantee(gnti) = distortion of central subspace U
    leg
        legend text associated with corresponding datum
    pst
        list of plot styles associated with `thetas`
    psm
        list of plot styles associated with `proj_dims`
    psk
        list of marker fill styles associated with `sub_dims`
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    """

    labels = [r'$\max\mathcal{D}_{\mathbf{A}}(\mathcal{U}^\prime)$',
              r'$\epsilon_{\mathcal{U}}$ s.t. '
              + r'$\mathcal{E}_\mathcal{T}(\epsilon_'
              + r'{\mathcal{U}},\theta_{\mathcal{T}}) = '
              + r'\mathcal{D}_{\mathbf{A}}(\mathcal{U})$']
    plot_scatter(ax, epsb, gnti, labels, leg, pst, psm, psk, txtopts, legopts)
    ax.yaxis.get_label().set_position((0, 1))
    ax.yaxis.get_label().set_horizontalalignment('right')
#    horizontalalignment='right', verticalalignment='bottom', **txtopts)
#    ax.yaxis.set_label_coords(-0.1, 1.)


# =============================================================================
# options
# =============================================================================


def default_options() -> (Styles, Styles, Styles,
                          Options, Options,
                          Sequence[float]):
    """
    Default options for plotting data

    Returns
    -------
    pmrks
        list of plot markers associated with `thetas`
    pcols
        list of plot colours associated with `proj_dims`
    pfills
        list of marker fill styles associated with `sub_dims`
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    siz
        (width, height) in inches
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['text.usetex'] = True
#    mpl.rcParams['text.latex.unicode'] = True
    mpl.rcParams['font.family'] = r'serif'

    siz = (7., 6.)
    # text style options for axes labels and legend
    txtopts = {'size': 32, 'family': 'serif'}
    # popsition of legend
    legpos = (1.5, 1)
    # legend options
    legopts = {'prop': {'size': 26, 'family': 'serif'},
               'bbox_to_anchor': legpos, 'handlelength': 1, 'frameon': False,
               'framealpha': 0., 'markerscale': 2}

    # list of plot markers associated with thetas
    pmrks = [{'marker': 'o'}, {'marker': '^'}, {'marker': 's'},
             {'marker': '*'}, {'marker': 'v'}]
    # list of plot colours associated with proj_dims
    pcols = [{'color': 'b'}, {'color': 'r'}, {'color': 'g'}]
    # list of marker fill styles associated with sub_dims
    pfills = [{'fillstyle': 'full'}, {'fillstyle': 'none'},
              {'fillstyle': ' top'}, {'fillstyle': 'bottom'},
              {'fillstyle': 'left'}, {'fillstyle': 'right'}]

    return pmrks, pcols, pfills, txtopts, legopts, siz


# =============================================================================
# running code
# =============================================================================


def load_and_plot(filename: str,
                  pmrks: Styles,
                  pcols: Styles,
                  pfills: Styles,
                  textopt: Options,
                  legopts: Options,
                  siz: Sequence[float] = (8., 6.)) -> Sequence[Figure]:
    """
    Load data from ``.npz`` file and plot

    Parameters
    ----------
    filename
        name of ``.npz`` file, w/o extension, with data
    pmrks
        list of plot markers associated with `thetas`
    pcols
        list of plot colours associated with `proj_dims`
    pfills
        list of marker fill styles associated with `sub_dims`
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    siz
        (width, height) in inches

    Returns
    -------
    figs
        list of figure objects, [fwd+legend, inv+legend, fwd, inv]
    """
    d = np.load(filename + '.npz')

    figs = [plt.figure(figsize=siz) for i in range(4)]
    axs = [fig.add_subplot(1, 1, 1) for fig in figs]

    funs = (plot_data, plot_data_inv) * 2
    xvars = ['eps', 'epsb'] * 2
    yvars = ['gnt', 'gnti'] * 2
    legs = (d['leg'].tolist(),) * 2 + (None,) * 2

    for fig, ax, fun, xv, yv, leg in zip(figs, axs, funs, xvars, yvars, legs):
        fun(ax, d[xv], d[yv], leg, pmrks, pfills, pcols, textopt, legopts)
        ax.grid(b=True)
        fig.tight_layout()

    d.close()

    return figs


def load_and_plot_and_save(filename: str,
                           pmrks: Styles,
                           pcols: Styles,
                           pfills: Styles,
                           figname: str,
                           textopt: Options,
                           legopts: Options,
                           siz: Sequence[float] = (8., 6.)):
    """
    Load data from ``.npz`` file, plot and save fig as ``.pdf`` file

    Parameters
    ----------
    filename
        name of ``.npz`` file, w/o extension, with data
    pmrks
        list of plot markers associated with `thetas` (dictionaries)
    pcols
        list of plot colours associated with `proj_dims` (dictionaries)
    pfills
        list of marker fill styles associated with `sub_dims`
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    figname
        stem of names of ``.pdf`` file, w/o extension, for saved figs
    siz
        (width, height) in inches
    """
    figs = load_and_plot(filename, pmrks, pcols, pfills, textopt, legopts, siz)

    suffixes = ['', '_inv', '_cropped', '_inv_cropped']
    for fig, suff in zip(figs, suffixes):
        fig.savefig(figname + suff + '.pdf', bbox_inches='tight')


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
