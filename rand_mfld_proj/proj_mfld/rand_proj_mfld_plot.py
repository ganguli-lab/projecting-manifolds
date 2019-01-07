# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 17:29:16 2016

@author: Subhy

Plot maximum distortion of Gaussian random manifolds under random projections

Functions
=========
make_fig_ax
    Make figure and axes objects, separate plots
make_fig_ax_2
    Make figure and axes objects, paired plots
plot_num_fig
    Plot graphs for numerics
plot_combo_figs
    Plot graphs for combined numerics and theory
plot_figs
    Plot all graphs
default_options
    dicts and tuples of default options for plots
load_and_fit
    load data and display results of linear fits
load_and_plot
    load data and plot figures
load_and_plot_and_save
    load data, plot figures and save pdfs
"""
import numpy as np
from numpy import ndarray as array
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools as it
from typing import Sequence, Mapping, Any, Optional
from . import rand_proj_mfld_theory as rpmt
from . import rand_proj_mfld_fit as rft

Styles = Sequence[Mapping[str, str]]
StyleSet = Mapping[str, Styles]
Options = Mapping[str, Any]
OptionSet = Mapping[str, Options]
Labels = Sequence[str]
LabelSet = Mapping[str, Labels]
Axes = mpl.axes.Axes
Figure = mpl.figure.Figure
Lines = Sequence[mpl.lines.Line2D]


# =============================================================================
# %%* pre-plotting
# =============================================================================


def leg_text(mfld_dims: Sequence[int],
             epsilons: Sequence[float],
             prefix: str = '',
             lgtxt: Optional[Labels] = None) -> Labels:  # legend entries
    """
    Make legend text entries

    Parameters
    ----------
    mfld_dim
        list of K, dimensionality of manifold
    epsilon
        list of epsilon, allowed distortion
    prefix
        string to prepend to each entry (default = '')
    lgtxt
        list of egend strings to append to (default = [])

    Returns
    -------
    lgtxt
        list of legend strings
    """
    if lgtxt is None:
        lgtxt = []

    if len(mfld_dims) == 1:
        lgtxt += [prefix + r'$\epsilon={0}$'.format(eps) for eps in epsilons]
#        for eps in epsilons:
#            lgtxt.append(prefix + r'$\epsilon={0}$'.format(eps))

    elif len(epsilons) == 1:
        lgtxt += [prefix + r'$K={0}$'.format(K) for K in mfld_dims]
#        for K in mfld_dims:
#            lgtxt.append(prefix + r'$K={0}$'.format(K))

    else:
        lgtxt += [prefix + r'$K={0}, \epsilon={1}$'.format(K, eps)
                  for (K, eps) in it.product(mfld_dims, epsilons)]
#        for K in mfld_dims:
#            for eps in epsilons:
#                lgtxt.append(prefix + r'$K={0}, \epsilon={1}$'.format(K, eps))

    return lgtxt


def make_fig_ax(num: int,
                siz: Sequence[float] = (8., 6.)) -> (Sequence[Figure],
                                                     Sequence[Axes]):
    """
    Make figure and axes objects, separate plots

    Parameters
    ----------
    num
        number of figs

    Returns
    -------
    figs
        list of figure objects
    axs
        list of axes objects
    """
    figs = [plt.figure(figsize=siz) for i in range(num)]
    axs = [fig.add_subplot(1, 1, 1) for fig in figs]
#    for gsi in range(num):
#        figs.append(plt.figure(figsize=siz))
#        axs.append(figs[-1].add_subplot(1, 1, 1))
    return figs, axs


def make_fig_ax_2(num: int,
                  siz: Sequence[float] = (16., 6.)) -> (Sequence[Figure],
                                                        Sequence[Axes]):
    """
    Make figure and axes objects, paired plots

    Parameters
    ----------
    num
        number of figs

    Returns
    -------
    figs
        list of figure objects
    axs
        list of axes objects
    """
    figs = [plt.figure(figsize=siz) for gsi in range(num)]
    axs = [fig.add_subplot(1, 2, j) for (fig, j) in it.product(figs, [1, 2])]
#    for gsi in range(num):
#        figs.append(plt.figure(figsize=siz))
#        axs.append(figs[-1].add_subplot(1, 2, 1))
#        axs.append(figs[-1].add_subplot(1, 2, 2))
    return figs, axs


# =============================================================================
# %%* plot part
# =============================================================================


def plot_vec(ax: Axes,
             NVs: array,
             Me_K: array,
             ph: Optional[Lines] = None, **kwargs) -> Lines:
    """
    Plot M \\epsilon^2 / K vs log N or log V / K

    Parameters
    ----------
    ax
        axes object to plot on
    NVs
        ndarray of N or V^1/K values
    Me_K
        ndarray of values of M \\epsilon^2 / K
    ph
        list of plot handles to append to (default = [])

    Returns
    -------
    ph
        list of plot handles
    """
    if ph is None:
        ph = []

    ph.append(ax.plot(np.log(NVs), Me_K, **kwargs)[0])

    return ph


def plot_vec_fit(ax: Axes,
                 NVs: array,
                 Me_K: array,
                 ph: Optional[Lines] = None,
                 marker: str = 'o',
                 fillstyle: str = 'full',
                 linestyle: str = 'solid', **kwargs) -> Lines:
    """
    Plot M \\epsilon^2 / K vs log N or log V / K

    Parameters
    ----------
    ax
        axes object to plot on
    NVs
        ndarray of N or V^1/K values
    Me_K
        ndarray of values of M \\epsilon^2 / K
    ph
        list of plot handles to append to (default = [])

    Returns
    -------
    ph
        list of plot handles
    """
    if ph is None:
        ph = []

    m, y, err = rft.linear_fit(np.log(NVs), Me_K)
    ax.plot(np.log(NVs), Me_K, marker=marker, fillstyle=fillstyle,
            linestyle='None', **kwargs)
    ph.append(ax.plot(np.log(NVs), y, linestyle=linestyle, **kwargs)[0])
#    print(m)
#    print(err)

    return ph


def plot_list(ax: Axes,
              NVs: array,
              Me_Ks: array,
              ph: Optional[Lines] = None,
              styles: Optional[Styles] = None,
              fit: bool = False,
              **kwargs) -> Lines:  # plot M*eps**2/K vs log N/V
    """
    Plot a list of M \\epsilon^2 / K vs log N or log V / K

    Parameters
    ----------
    ax
        axes object to plot on
    NVs
        ndarray of N or V^1/K values
    Me_Ks
        list of ndarrays of values of M \\epsilon^2 / K
    ph
        list of plot handles to append to (default = [])
    styles
        list of dicts of plot style options for each member of MeKs
    fit : bool
        Plot linear fits if True, Join points if False.

    Returns
    -------
    ph
        list of plot handles
    """
    if ph is None:
        ph = []

    if fit:
        plotfn = plot_vec_fit
    else:
        plotfn = plot_vec

    if styles is None:
        styles = ({},) * len(Me_Ks)

    if len(NVs) == len(Me_Ks):
        eXs = NVs
    else:
        eXs = (NVs,) * len(Me_Ks)

    for NV, Me_K, style in zip(eXs, Me_Ks, styles):
        ph = plotfn(ax, NV, Me_K, ph, **style, **kwargs)

    return ph


def plot_all(ax: Axes,
             num_Xs: array,
             M_num: array,
             th_Xs: array,
             M_thr: Sequence[array],
             epsilons: array,
             Ks: Sequence[int],
             xlabel: str,
             labels: Optional[LabelSet],
             opts: OptionSet,
             styleK: StyleSet,
             styleF: Styles,
             fit: bool = False,
             **kwargs):  # plot Me^2/K vs log N,V for all
    """
    Plot M \\epsilon^2 / K vs log N or log V / K for numerics and all theories

    Parameters
    ----------
    ax
        axes object to plot on
    num_Xs
        ndarray of N or V^1/K values for numerics
    M_num
        M \\epsilon^2 / K.
        numerical M's: required projection dimensionality (#K,#epsilon,#NV)
    th_Xs
        ndarray of N or V^1/K values for theory
    M_thr
        tuple of ndarrays of theoretical (M*epsilon**2/K)'s (4)
        tuple members:

        LGG
            Our theoretical M's (#K,#epsilon,#NV)
        BW
            Baraniuk & Wakin (2009) theoretical M's, (#K,#epsilon,#NV)
        Vr
            Verma (2011) theoretical M's, (#K,#epsilon,#NV)
        EW
            Eftekhari & Wakin (2015) theoretical M's, (#K,#epsilon,#NV)
    epsilons
        list of allowed distortions
    Ks
        list of intrinsic dimensionalities
    xlabel
        x-axis label (log N or log V)
    labels
        dict of strings for legends, or None if no legend.
        **labels[key]** = list of strings: [legend (short), title (long)]

        labels['Num']
            numerical results
        labels['LGG']
            our theory
        labels['BW']
            Baraniuk & Wakin theory
        labels['Vr']
            Verma theory
        labels['Vr']
            Verma theory
    opts
        dict of dicts of options for plots

        opts['tx']
            text options for x,y-axis label and title
        opts['lg']
            options for legends
    StyleK
        dict of lists of dicts of plot style options

        StyleK['num']
            list of dicts of style plot options for numerical plots (#K,)
        StyleK['thr']
            list of dicts of style plot options for theoretical plots (#K,)
    StyleF
        list of dicts of plot style options for each sim/theory (>1+#theories,)
    fit : bool
        Plot linear fits if True, Join points if False.
    """
    ind = 0

    if 'prop' in opts['lg'] and 'size' in opts['lg']['prop']:
        ax.tick_params(axis='both', which='major',
                       labelsize=opts['lg']['prop']['size'])

    ph = []

    ph = plot_list(ax, num_Xs, list(M_num[:, ind, :]),
                   ph, styleK['num'], fit=fit, **styleF[0], **kwargs)
    for M_t, stlF in zip(M_thr, styleF[1:]):
        ph = plot_list(ax, th_Xs, list(M_t[:, ind, :]),
                       ph, styleK['thr'], **stlF, **kwargs)

    if labels:
        leg = []
        leg = leg_text(Ks, [1], labels['Num'][0], leg)
        labs = ['LGG', 'BW', 'Vr', 'EW']
        leg += [labels[lab][0] for lab in labs]
        phl = ph[1::2]
        phl.insert(0, ph[0])
        ax.legend(phl, leg, **opts['lg'])

    ax.set_xlabel(xlabel, **opts['tx'])
    ax.set_ylabel(r'$M^* \epsilon^2 / K$', **opts['tx'])
    ax.set_yscale('log')


def plot_one(ax: Axes,
             Xs: array,
             Mes: array,
             epsilons: array,
             Ks: Sequence[int],
             xlabel: str,
             title: str,
             opts: OptionSet,
             styleK: StyleSet,
             styleE: Styles,
             leg: Optional[LabelSet] = None,
             fit: bool = False,
             **kwargs):  # plot Me^2/K vs log N,V for one sim/theory
    """
    Plot M \\epsilon^2 / K vs log N or log V / K for one theory or numerics

    Parameters
    ----------
    ax
        axes object to plot on
    Xs
        ndarray of N or V^1/K values for theory
    Mes
        M \\epsilon^2 / K.
        M's: required projection dimensionality, (#K,#epsilon,#NV)
    epsilons
        list of allowed distortions
    Ks
        list of intrinsic dimensionalities
    xlabel
        x-axis label (log N or log V)
    title
        string for title
    opts
        dict of dicts of options for plots

        opts['tx']
            text options for x,y-axis label and title
        opts['lg']
            options for legends
    StyleK
        list of dicts of plot style options for eack K  (#K,)
    StyleE
        list of dicts of plot style options for each epsilon (>#epsilon,)
    leg
        empty list for legend entries, None if no legend
    fit : bool
        Plot linear fits if True, Join points if False.
    """
    if 'prop' in opts['lg'] and 'size' in opts['lg']['prop']:
        ax.tick_params(axis='both', which='major',
                       labelsize=opts['lg']['prop']['size'])

    ph = []
    ph = plot_list(ax, Xs[0], list(Mes[0, :, :]), ph,
                   styleE, fit=fit, **styleK[0], **kwargs)
    ph = plot_list(ax, Xs[1], list(Mes[1, :, :]), ph,
                   styleE, fit=fit, **styleK[1], **kwargs)

    ax.set_xlabel(xlabel, **opts['tx'])
    ax.set_ylabel(r'$M^* \epsilon^2 / K$', **opts['tx'])
    ax.set_title(title, **opts['tx'])

    if leg is not None:
        leg = leg_text(Ks, epsilons, lgtxt=leg)
        ax.legend(ph, leg, **opts['lg'])


# =============================================================================
# %%* plot full fig
# =============================================================================


def read_data(fileobj):
    """
    """
    prob = fileobj['prob']

    nums = fileobj['ambient_dims']
    vols = fileobj['vols']

    Ks = np.arange(1, 1+len(vols))[..., None, None]
    epsilons = fileobj['epsilons']
#    Mes_num_N = fileobj['num_N'] * epsilons[..., None]**2 / Ks
#    Mes_num_V = fileobj['num_V'] * epsilons[..., None]**2 / Ks
    Mes_num_N = fileobj['M_num'][..., -1, :] * epsilons[..., None]**2 / Ks
    Mes_num_V = fileobj['M_num'][..., -1] * epsilons[..., None]**2 / Ks

    return Mes_num_N, Mes_num_V, nums, vols, Ks.squeeze(), epsilons, prob


def plot_num_figs(axs: Sequence[Axes],
                  fileobj: np.lib.npyio.NpzFile,
                  opts: OptionSet,
                  labels: LabelSet,
                  styleK: StyleSet,
                  styleE: Styles,
                  fit: bool = True,
                  **kwargs):
    """
    axs
        list of axes objects
    fileobj
        instance of NpzFile class from .npz file with data, with fields:

        num_N
            values of M when varying N, ndarray (#K,#epsilon,#N)
        num_V
            values of M when varying V, ndarray (#K,#epsilon,#N)
        prob
            allowed failure probability
        ambient_dims
            list of N's, dimensionality of ambient space, ((#N,), 1)
            tuple for varying N and V, second one a scalar
        vols_N
             tuple of V^1/K, for each K,
        vols_V
            tuple of V^1/K, for each K, each member is an ndarray (#V)
        epsilons
            list of allowed distortions (#epsilon)
    opts
        dict of dicts of options for plots

        opts['tx']
            text options for x,y-axis label and title
        opts['lg']
            options for legends
    labels
        dict of strings for legends.
        **labels[key]** = list of strings: [legend (short), title (long)]

        labels['Num']
            numerical results
        labels['LGG']
            our theory
        labels['BW']
            Baraniuk & Wakin theory
        labels['Vr']
            Verma theory
        labels['Vr']
            Verma theory
    StyleK
        dict of lists of dicts of plot style options

        StyleK['num']
            list of dicts of style plot options for numerical plots (#K,)
        StyleK['thr']
            list of dicts of style plot options for theoretical plots (#K,)
    StyleE
        list of dicts of plot style options for each epsilon (>#e,)
    """
    Mes_num_N, Mes_num_V, nums, vols, Ks, epsilons, prob = read_data(fileobj)
    numt = (nums,) * len(Ks)

    nlab = r'$\ln N$'
    vlab = r'$(\ln\mathcal{V})/K$'

    plot_one(axs[0], numt, Mes_num_N, epsilons, Ks, nlab, labels['Num'][1],
             opts, styleK['num'], styleE, [], fit=fit, **kwargs)
    plot_one(axs[1], vols, Mes_num_V, epsilons, Ks, vlab, labels['Num'][1],
             opts, styleK['num'], styleE, fit=fit, **kwargs)


def plot_thr_figs(axs: Sequence[Axes],
                  fileobj: np.lib.npyio.NpzFile,
                  opts: OptionSet,
                  labels: LabelSet,
                  styleK: StyleSet,
                  styleE: Styles,
                  **kwargs):
    """
    axs
        list of axes objects
    fileobj
        instance of NpzFile class from .npz file with data, with fields:

        num_N
            values of M when varying N, ndarray (#K,#epsilon,#N)
        num_V
            values of M when varying V, ndarray (#K,#epsilon,#N)
        prob
            allowed failure probability
        ambient_dims
            list of N's, dimensionality of ambient space, ((#N,), 1)
            tuple for varying N and V, second one a scalar
        vols_N
             tuple of V^1/K, for each K,
        vols_V
            tuple of V^1/K, for each K, each member is an ndarray (#V)
        epsilons
            list of allowed distortions (#epsilon)
    opts
        dict of dicts of options for plots

        opts['tx']
            text options for x,y-axis label and title
        opts['lg']
            options for legends
    labels
        dict of strings for legends.
        **labels[key]** = list of strings: [legend (short), title (long)]

        labels['Num']
            numerical results
        labels['LGG']
            our theory
        labels['BW']
            Baraniuk & Wakin theory
        labels['Vr']
            Verma theory
        labels['Vr']
            Verma theory
    StyleK
        dict of lists of dicts of plot style options

        StyleK['num']
            list of dicts of style plot options for numerical plots (#K,)
        StyleK['thr']
            list of dicts of style plot options for theoretical plots (#K,)
    StyleE
        list of dicts of plot style options for each epsilon (>#e,)
    """
    nums, vols, Ks, epsilons, prob = read_data(fileobj)[2:]

    theory = rpmt.get_all_analytic(epsilons, nums, vols, prob)

    Ns = (theory[0],) * len(Ks)
    Vs = (theory[1],) * len(Ks)

    nlab = r'$\ln N$'
    vlab = r'$(\ln\mathcal{V})/K$'

    labs = ['LGG', 'BW', 'Vr', 'EW']
    for ax_n, ax_v, lab, M_N, M_V in zip(axs[::2], axs[1::2], labs,
                                         theory[2::2], theory[3::2]):
        plot_one(ax_n, Ns, M_N, epsilons, Ks, nlab,
                 labels[lab][1], opts, styleK['thr'], styleE, **kwargs)
        plot_one(ax_v, Vs, M_V, epsilons, Ks, vlab,
                 labels[lab][1], opts, styleK['thr'], styleE, **kwargs)


def plot_combo_figs(axs: Sequence[Axes],
                    fileobj: np.lib.npyio.NpzFile,
                    opts: OptionSet,
                    labels: LabelSet,
                    styleK: StyleSet,
                    styleF: Styles,
                    **kwargs):
    """
    axs
        list of axes objects
    fileobj
        instance of NpzFile class from .npz file with data, with fields:

        num_N
            values of M when varying N, ndarray (#K,#epsilon,#N)
        num_V
            values of M when varying V, ndarray (#K,#epsilon,#N)
        prob
            allowed failure probability
        ambient_dims
            list of N's, dimensionality of ambient space, ((#N,), 1)
            tuple for varying N and V, second one a scalar
        vols_N
             tuple of V^1/K, for each K,
        vols_V
            tuple of V^1/K, for each K, each member is an ndarray (#V)
        epsilons
            list of allowed distortions (#epsilon)
    opts
        dict of dicts of options for plots

        opts['tx']
            text options for x,y-axis label and title
        opts['lg']
            options for legends
    labels
        dict of strings for titles and legends.
        **labels[key]** = list of strings: [legend (short), title (long)].
        The keys are:
            Num
                numerical results
            LGG
                our theory
            BW
                Baraniuk & Wakin  theory
            Vr
                Verma theory
            EW
                Eftekhari & Wakin theory
    StyleK
        dict of lists of dicts of plot style options

        StyleK['num']
            list of dicts of style plot options for numerical plots (#K,)
        StyleK['thr']
            list of dicts of style plot options for theoretical plots (#K,)
    StyleF
        list of dicts of plot style options for each sim/theory (>1+#theories,)
    """
    Mes_num_N, Mes_num_V, nums, vols, Ks, epsilons, prob = read_data(fileobj)
    numt = (nums,) * len(Ks)

    theory = rpmt.get_all_analytic(epsilons, nums, vols, prob)

    nlab = r'$\ln N$'
    vlab = r'$(\ln\mathcal{V})/K$'

    plot_all(axs[0], numt, Mes_num_N, theory[0], theory[2::2],
             epsilons, Ks, nlab, labels, opts, styleK, styleF, **kwargs)
    plot_all(axs[1], vols, Mes_num_V, theory[1], theory[3::2],
             epsilons, Ks, vlab, None, opts, styleK, styleF, **kwargs)


def plot_all_figs(axs: Sequence[Axes],
                  fileobj: np.lib.npyio.NpzFile,
                  opts: OptionSet,
                  labels: LabelSet,
                  styleK: StyleSet,
                  styleE: Styles,
                  styleF: Styles,
                  fit: bool = False,
                  **kwargs):
    """
    axs
        list of axes objects
    fileobj
        instance of NpzFile class from .npz file with data, with fields:

        num_N
            values of M when varying N, ndarray (#K,#epsilon,#N)
        num_V
            values of M when varying V, ndarray (#K,#epsilon,#N)
        prob
            allowed failure probability
        ambient_dims
            list of N's, dimensionality of ambient space, ((#N,), 1)
            tuple for varying N and V, second one a scalar
        vols_N
             tuple of V^1/K, for each K,
        vols_V
            tuple of V^1/K, for each K, each member is an ndarray (#V)
        epsilons
            list of allowed distortions (#epsilon)
    opts
        dict of dicts of options for plots

        opts['tx']
            text options for x,y-axis label and title
        opts['lg']
            options for legends
    labels
        dict of strings for titles and legends.
        **labels[key]** = list of strings: [legend (short), title (long)].
        The keys are:
            Num
                numerical results
            LGG
                our theory
            BW
                Baraniuk & Wakin  theory
            Vr
                Verma theory
            EW
                Eftekhari & Wakin theory
    StyleK
        dict of lists of dicts of plot style options

        StyleK['num']
            list of dicts of style plot options for numerical plots (#K,)
        StyleK['thr']
            list of dicts of style plot options for theoretical plots (#K,)
    StyleE
        list of dicts of plot style options for each epsilon (>#e,)
    StyleF
        list of dicts of plot style options for each sim/theory (>1+#theories,)
    fit : bool
        Plot linear fits if True, Join points if False.
    """
    plot_combo_figs(axs[:2], fileobj, opts, labels, styleK, styleF, **kwargs)
    plot_num_figs(axs[2:4], fileobj, opts, labels, styleK, styleE, **kwargs)
    plot_thr_figs(axs[4:], fileobj, opts, labels, styleK, styleE, **kwargs)


# =============================================================================
# %%* options
# =============================================================================


def default_options() -> (OptionSet, LabelSet,
                          StyleSet, Styles, Styles):
    """
    Default options for plotting data

    Returns
    -------
    opts
        dict of dicts of options for plots

        opts['tx']
            text options for x,y-axis label and title
        opts['lg']
            options for legends
    labels
        dict of strings for titles and legends.
        **labels[key]** = list of strings: [legend (short), title (long)].
        The keys are:
            Num
                numerical results
            LGG
                our theory
            BW
                Baraniuk & Wakin  theory
            Vr
                Verma theory
            EW
                Eftekhari & Wakin theory
    StyleK
        dict of lists of dicts of plot style options

        StyleK['num']
            list of dicts of style plot options for numerical plots (#K,)
        StyleK['thr']
            list of dicts of style plot options for theoretical plots (#K,)
    StyleE
        list of dicts of plot style options for each epsilon (>#e,)
    StyleF
        list of dicts of plot style options for each sim/theory (>1+#theories,)

    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['text.usetex'] = True
#    mpl.rcParams['text.latex.unicode'] = True
    mpl.rcParams['font.family'] = r'serif'

    txtopts = {'size': 30, 'family': 'serif'}
    lgprops = {'prop': {'size': 'xx-large', 'family': 'serif'},
               'numpoints': 1, 'loc': 'upper right'}

    Numlab = ['Sim: ', 'Simulation']
    LGGlab = ['New theory', 'Current paper']
    BWlab = ['BW theory', r'Baraniuk and Wakin, (2009)']
    Vrlab = ['Verma theory', r'Verma (2011)']
    EWlab = ['EW theory', r'Eftekhari and Wakin (2015)']

    styleNum = [{'marker': 'o', 'fillstyle': 'full', 'linestyle': 'dashed'},
                {'marker': '^', 'fillstyle': 'full', 'linestyle': 'solid'}]
    styleThr = [{'linestyle': 'dashed'}, {'linestyle': 'solid'}]
    styleK = {'num': styleNum, 'thr': styleThr}
    # bgm
    styleE = [{'color': (0.149019607843137,
                         0.545098039215686,
                         0.823529411764706)},
              {'color': (0.521568627450980,
                         0.600000000000000,
                         0.)},
              {'color': (0.827450980392157,
                         0.211764705882353,
                         0.509803921568627)}]
    # yvtorc
    styleF = [{'color': (0.709803921568628,
                         0.537254901960784,
                         0.)},
              {'color': (0.423529411764706,
                         0.443137254901961,
                         0.768627450980392)},
              {'color': (0.,
                         0.501960784313726,
                         0.501960784313726)},
              {'color': (0.796078431372549,
                         0.294117647058824,
                         0.0862745098039216)},
              {'color': (0.196078431372549,
                         0.662745098039216,
                         0.184313725490196)},
              {'color': (0.164705882352941,
                         0.631372549019608,
                         0.596078431372549)}]

    opts = {'tx': txtopts, 'lg': lgprops}
    labels = {'Num': Numlab, 'LGG': LGGlab, 'BW': BWlab, 'Vr': Vrlab,
              'EW': EWlab}

    return opts, labels, styleK, styleE, styleF

# =============================================================================
# %%* running code
# =============================================================================


def load_and_fit(filename: str, ix: Optional[array] = None):
    """
   Load data from .npz file and display linear fit

    Parameters
    ----------
    filename
        name of .npz file, w/o extension, for data, with fields:

        num_N
            values of M when varying N, ndarray (#K,#epsilon,#N)
        num_V
            values of M when varying V, ndarray (#K,#epsilon,#N)
        prob
            allowed failure probability
        ambient_dims
            list of N's, dimensionality of ambient space, ((#N,), 1)
            tuple for varying N and V, second one a scalar
        vols_N
             tuple of V^1/K, for each K,
        vols_V
            tuple of V^1/K, for each K, each member is an ndarray (#V)
        epsilons
            list of allowed distortions (#epsilon)
    ix
        ndarray of indices of variables to include in linear regression.
        chosen from: const, ln K, -ln e, ln N, ln V
    """
    d = np.load(filename + '.npz')

    rft.dsp_multi(d, ix)
    d.close()


def load_and_plot(filename: str,
                  opts: OptionSet,
                  labels: LabelSet,
                  styleK: StyleSet,
                  styleE: Styles,
                  styleF: Styles,
                  **kwds) -> Sequence[Figure]:  # load data and plot
    """
    Load data from .npz file and plot

    Parameters
    ----------
    filename
        name of .npz file, w/o extension, for data, with fields:

        num_N
            values of M when varying N, ndarray (#K,#epsilon,#N)
        num_V
            values of M when varying V, ndarray (#K,#epsilon,#N)
        prob
            allowed failure probability
        ambient_dims
            list of N's, dimensionality of ambient space, ((#N,), 1)
            tuple for varying N and V, second one a scalar
        vols_N
             tuple of V^1/K, for each K,
        vols_V
            tuple of V^1/K, for each K, each member is an ndarray (#V)
        epsilons
            list of allowed distortions (#epsilon)
    opts
        dict of dicts of options for plots

        opts['tx']
            text options for x,y-axis label and title
        opts['lg']
            options for legends
    labels
        dict of strings for titles and legends.
        **labels[key]** = list of strings: [legend (short), title (long)].
        The keys are:
            Num
                numerical results
            LGG
                our theory
            BW
                Baraniuk & Wakin  theory
            Vr
                Verma theory
            EW
                Eftekhari & Wakin theory
    StyleK
        dict of lists of dicts of plot style options

        StyleK['num']
            list of dicts of style plot options for numerical plots (#K,)
        StyleK['thr']
            list of dicts of style plot options for theoretical plots (#K,)
    StyleE
        list of dicts of plot style options for each epsilon (>#e,)
    StyleF
        list of dicts of plot style options for each sim/theory (>1+#theories,)

    Returns
    -------
    figs
        list of figure objects
    """

#    figs, axs = make_fig_ax_2(6)
    figs, axs = make_fig_ax_2(2)

    d = np.load(filename + '.npz')

    plot_num_figs(axs[:2], d, opts, labels, styleK, styleE, fit=False, **kwds)
    plot_combo_figs(axs[2:], d, opts, labels, styleK, styleF, **kwds)
#    plot_all_figs(axs, d, opts, labels, styleK, styleE, styleF, **kwargs)
#    rft.dsp_multi(d)
    d.close()
    return figs


def load_and_plot_and_save(filename: str,
                           opts: OptionSet,
                           labels: LabelSet,
                           fignames: str,
                           figpath: str,
                           styleK: StyleSet,
                           styleE: Styles,
                           styleF: Styles,
                           **kwargs):  # load data and plot
    """
    Load data from .npz file and plot

    Parameters
    ----------
    filename
        name of .npz file, w/o extension, for data, with fields:

        num_N
            values of M when varying N, ndarray (#K,#epsilon,#N)
        num_V
            values of M when varying V, ndarray (#K,#epsilon,#N)
        prob
            allowed failure probability
        ambient_dims
            list of N's, dimensionality of ambient space, ((#N,), 1)
            tuple for varying N and V, second one a scalar
        vols_N
             tuple of V^1/K, for each K,
        vols_V
            tuple of V^1/K, for each K, each member is an ndarray (#V)
        epsilons
            list of allowed distortions (#epsilon)
    opts
        dict of dicts of options for plots

        opts['tx']
            text options for x,y-axis label and title
        opts['lg']
            options for legends
    labels
        dict of strings for titles and legends.
        **labels[key]** = list of strings: [legend (short), title (long)].
        The keys are:
            Num
                numerical results
            LGG
                our theory
            BW
                Baraniuk & Wakin  theory
            Vr
                Verma theory
            EW
                Eftekhari & Wakin theory
    fignames
        list of .pdf file names, w/o extensions or paths
    figpath
        path to folder for .pdf files, ending with '/'
    StyleK
        dict of lists of dicts of plot style options

        StyleK['num']
            list of dicts of style plot options for numerical plots (#K,)
        StyleK['thr']
            list of dicts of style plot options for theoretical plots (#K,)
    StyleE
        list of dicts of plot style options for each epsilon (>#e,)
    StyleF
        list of dicts of plot style options for each sim/theory (>1+#theories,)
    """

    figs, axs = make_fig_ax(12)

    d = np.load(filename + '.npz')

    plot_all_figs(axs, d, opts, labels, styleK, styleE, styleF, **kwargs)

    for fig, figname in zip(figs, fignames):
        fig.savefig(figpath + figname + '.pdf', bbox_inches='tight')

    d.close()


# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
