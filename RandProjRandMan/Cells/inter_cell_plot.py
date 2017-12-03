# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:14:44 2016

@author: Subhy

Plot disortion of vectors between cell centres and vectors between edges of
balls that enclose cells, to test assertion that:

.. math::
    D_A(x) < E_C(\epsilon,\\theta_C) \implies D_A(y) < \epsilon
                                            \;\\forall y \in C
| where C = chordal cone,
| :math:`\\theta_C` = angle between centre and edge,
| x = central vector of cone.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Sequence, Mapping, Any

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

    Parameters
    ==========
    ax
        axes object for plotting on
    """
    xl = ax.get_xlim()
    yl = ax.get_ylim()

    lim = np.array([np.maximum(xl[0], yl[0]), np.minimum(xl[1], yl[1])])

    ax.plot(lim, lim, 'k-')
    ax.set_xlim(xl)
    ax.set_ylim(yl)


def plot_scatter(ax: Axes,
                 eps: np.ndarray,
                 gnt: np.ndarray,
                 labels: Labels,
                 leg: Labels,
                 pst: Styles,
                 psm: Styles,
                 txtopts: Options,
                 legopts: Options):
    """
    Plot all data and legend

    Parameters
    ==========
    ax
        axes object for plotting on
    eps
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
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    """

    if 'prop' in legopts and 'size' in legopts['prop']:
        ax.tick_params(axis='both', which='major',
                       labelsize=legopts['prop']['size'])

    lhs = []

    for i, pt in enumerate(pst[:eps.shape[0]]):
        for j, pm in enumerate(psm[:eps.shape[1]]):
            lhs.append(ax.plot(eps[i, j], gnt[i, j],
                               **pt, **pm, linestyle='none')[0])
        # extra element at end of each row: label with value of theta
        lhs.append(ax.plot(eps[i, 0], gnt[i, 0],
                           **pt, **psm[0], linestyle='none')[0])

    ax.set_xlabel(labels[0], **txtopts)
    ax.set_ylabel(labels[1], **txtopts)
#    ax.yaxis.set_label_coords(-0.1, 0.5)

    if leg is not None:
        hstep = gnt.shape[1]
        lhs = lhs[0:hstep] + lhs[hstep::(hstep + 1)]
        leg = leg[0:hstep] + leg[hstep::(hstep + 1)]
        ax.legend(lhs, leg, numpoints=1, **legopts)

    plot_equality(ax)


def plot_data(ax: Axes,
              epsx: np.ndarray,
              gnt: np.ndarray,
              leg: Labels,
              pst: Styles,
              psm: Styles,
              txtopts: Options,
              legopts: Options):  # plot all data
    """
    Plot all data and legend

    Parameters
    ==========
    ax
        axes object for plotting on
    epsx
        distortion of x
    gnt
        guarantee(maximum distortion of y) for y in chordal cone
    leg
        legend text associated with corresponding datum
    pst
        list of plot styles associated with `thetas`
    psm
        list of plot styles associated with `proj_dims`
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    """

    labels = [r'$\mathcal{D}_{\mathbf{A}}(\mathbf{x})$',
              r'$\mathcal{E}_\mathcal{C}(\max\mathcal{D}_' +
              r'{\mathbf{A}}(\mathbf{y}),\theta_{\mathcal{C}})$']
    plot_scatter(ax, epsx, gnt, labels, leg, pst, psm, txtopts, legopts)
#    ax.yaxis.set_label_coords(-0.1, 0.5)


def plot_data_inv(ax: Axes,
                  epsy: np.ndarray,
                  gnti: np.ndarray,
                  leg: Labels,
                  pst: Styles,
                  psm: Styles,
                  txtopts: Options,
                  legopts: Options):
    """plot all data

    Plot all data and legend

    Parameters
    ==========
    ax
        axes object for plotting on
    epsy
        maximum distortion of y for y in chordal cone
    gnti
        guarantee(gnti) = distortion of x
    leg
        legend text associated with corresponding datum
    pst
        list of plot styles associated with thetas (dictionaries)
    psm
        list of plot styles associated with proj_dims (dictionaries)
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    """

    labels = [r'$\max\mathcal{D}_{\mathbf{A}}(\mathbf{y})$',
              r'$\epsilon_{\mathbf{x}}$ s.t. ' +
              r'$\mathcal{E}_\mathcal{C}(\epsilon_{\mathbf{x}}, ' +
              r'\theta_{\mathcal{C}}) = ' +
              r'\mathcal{D}_{\mathbf{A}}(\mathbf{x})$']
    plot_scatter(ax, epsy, gnti, labels, leg, pst, psm, txtopts, legopts)
    ax.yaxis.get_label().set_position((0, 1))
    ax.yaxis.get_label().set_horizontalalignment('right')
#          horizontalalignment='right', verticalalignment='bottom', **txtopts)
#    ax.yaxis.set_label_coords(-0.1, 1.)


# =============================================================================
# options
# =============================================================================


def default_options() -> (Styles, Styles, Options, Options, Sequence[float]):
    """
    Default options for plotting data

    Returns
    =======
    pmrks
        list of plot markers associated with `thetas`
    pcols
        list of plot colours associated with `proj_dims`
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    siz
        (width, height) in inches
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode'] = True
    mpl.rcParams['font.family'] = r'serif'

    siz = (7., 6.)
    # text style options for axes labels and legend
    txtopts = {'size': 34, 'family': 'serif'}
    # position of legend
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

    return pmrks, pcols, txtopts, legopts, siz


# =============================================================================
# running code
# =============================================================================


def load_and_plot(filename: str,
                  pmrks: Styles,
                  pcols: Styles,
                  textopt: Options,
                  legopts: Options,
                  siz: Sequence[float]=(8., 6.)) -> Sequence[Figure]:
    """
    Load data from ``.npz`` file and plot

    Parameters
    ==========
    filename
        name of ``.npz`` file, w/o extension, with data
    pmrks
        list of plot markers associated with `thetas`
    pcols
        list of plot colours associated with `proj_dims`
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    siz
        (width, height) in inches

    Returns
    =======
    figs
        list of figure objects, [fwd+legend, inv+legend, fwd, inv]
    """
    d = np.load(filename + '.npz')

    figs = [plt.figure(figsize=siz) for i in range(4)]
    axs = [fig.add_subplot(1, 1, 1) for fig in figs]

    funs = (plot_data, plot_data_inv) * 2
    xvars = ['epsx', 'epsy'] * 2
    yvars = ['gnt', 'gnti'] * 2
    legs = (d['leg'].tolist(),) * 2 + (None,) * 2

    for fig, ax, fun, xv, yv, leg in zip(figs, axs, funs, xvars, yvars, legs):
        fun(ax, d[xv], d[yv], leg, pmrks, pcols, textopt, legopts)
        ax.grid(b=True)
        fig.tight_layout()

    d.close()

    return figs


def load_and_plot_and_save(filename: str,
                           pmrks: Styles,
                           pcols: Styles,
                           figname: str,
                           textopt: Options,
                           legopts: Options,
                           siz: Sequence[float]=(8., 6.)):
    """
    Load data from ``.npz`` file, plot and save fig as ``.pdf`` file

    Parameters
    ==========
    filename
        name of ``.npz`` file, w/o extension, with data
    pmrks
        list of plot markers associated with `thetas` (dictionaries)
    pcols
        list of plot colours associated with `proj_dims` (dictionaries)
    txtopts
        text style options for axes labels
    legopts
        style options for legend
    figname
        stem of names of ``.pdf`` file, w/o extension, for saved figs
    siz
        (width, height) in inches
    """
    figs = load_and_plot(filename, pmrks, pcols, textopt, legopts, siz)

    figs[0].savefig(figname + '.pdf', bbox_inches='tight')
    figs[1].savefig(figname + '_inv.pdf', bbox_inches='tight')
    figs[2].savefig(figname + '_cropped.pdf', bbox_inches='tight')
    figs[3].savefig(figname + '_inv_cropped.pdf', bbox_inches='tight')


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
