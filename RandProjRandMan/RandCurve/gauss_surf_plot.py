# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:46:16 2016

@author: Subhy

Plot distance, principal angles between tangent spaces and curvature as a
function of position on a Gaussian random surface in a high dimensional space

Functions
=========
make_fig_ax
    Make figure and axes objects
plot_data
    Plot all graphs for one of distance, angle, curvature
default_options
    dicts and tuples of default options for plots
load_and_plot
    load data and plot figures
load_and_plot_and_save
    load data, plot figures and save pdfs
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# plotting
# =============================================================================


def make_fig_ax(siz, num_ax):  # make figure and axes objects
    """
    Make figure and axes objects

    Returns
    -------
    fig, ax
        figure object, axes object

    Parameters
    ----------
    siz
        (width, height) in inches
    num_ax
        number of axes in figure
    """
    fig = plt.figure(figsize=siz)
    gspecs = plt.GridSpec(1, num_ax, width_ratios=[1] * num_ax)
    ax = []
    for gsi in range(num_ax):
        ax.append(fig.add_subplot(gspecs[0, gsi]))
    return fig, ax


def common_colorbar(imh, axh, labtext, labopt):  # add colorbar to axes
    """
    Add colorbar to collection of axes

    Parameters
    ----------
    imh
        pcolormesh object for heatmap with color info
    axh
        axes object to place colorbar on
    labtext
        string for colorbar label
    labopt
        options for colorbar label text
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axh)
    cax = divider.append_axes("right", "10%", pad="20%")
    cbh = plt.colorbar(imh, cax=cax)
    cbh.set_label(labtext, **labopt)


def common_clim(imh, cmin=0.0):  # set all clims equal
    """
    Make the clim for each image in list imh the same

    Parameters
    ----------
    imh
        list of pcolormesh objects with heatmaps
    cmin
        lower end of clim
    """
    cmax = 1.0 * cmin
    for im in imh:
        cl = im.get_clim()
        cmax = np.maximum(cmax, cl[1])

    for im in imh:
        im.set_clim((cmin, cmax))


def make_heatmaps(axh, x, y, dataa, xylabl, cblabl, titl, opts, lpad=27,
                  sample=1):  # make set of heat maps
    """
    Make set of heat maps

    Parameters
    ----------
    axh
        list of axes objects
    x,y
        ndarray of sigma^a positions
    dataa
        list of ndarrays of heatmap data
    xylabl
        list of strings: [x-axis label, y-axis label]
    cblabl
        list of strings for colorbar label [name, formula] of quantity
    titl
        list of strings for heatmap titles
    opts
        dictionary of options

        opts['im']
            options for pcolormesh for heatmaps
        opts['asp']
            aspect ratio of pcolormesh for heatmaps
        opts['tx']
            text options for x,y-axis label and title
        opts['cb']
            text options for colorbar label
    lpad
        padding for colorbar label
    sample
        plot every sample'th point
    """
    if 'prop' in opts['lg'] and 'size' in opts['lg']['prop']:
        labelsize = opts['lg']['prop']['size']
    imh = []
    for ax, dat, tit in zip(axh, dataa, titl):
        ax.tick_params(axis='both', which='major',
                       labelsize=labelsize)
        if x.shape[0] == dat.shape[0] + 1:
            imh.append(ax.pcolormesh(x[::sample], y[::sample],
                                     dat[::sample, ::sample].T, **opts['im']))
        else:
            imh.append(ax.pcolormesh(x[::2*sample], y[::2*sample],
                                     dat[::sample, ::sample].T, **opts['im']))
        imh[-1].set_edgecolor('face')
        ax.set_xlabel(xylabl[0], **opts['tx'])
        ax.set_ylabel(xylabl[1], labelpad=-8, **opts['tx'])
        ax.set_title(tit, **opts['tx'])
        ax.set_aspect(opts['asp'])

    opts['cb']['labelpad'] = lpad
    common_clim(imh)
#    common_colorbar(imh[0], axh[0], cblabl[0] + ', ' + cblabl[1], opts['cb'])
    common_colorbar(imh[0], axh[0], cblabl[1], opts['cb'])


def make_scatter(ax, x, y, ldata, titles, opts, sample=4):  # Make scatter plot
    """
    Make scatter plot of comparison of theory & expt

    Parameters
    ----------
    ax
        axes objectt
    x
        ndarray of rho values
    y
        list of ndarrays of simulation values
    ldata
        tuple of ndarrays, (x,y) of values for theory line on scatter plot
    titles
        list of strings [name, formula(e)] of quantity(ies)
    opts
        dictionary of options

        opts['tx']
            text options for x,y-axis label and title
        opts['lg']
            options for legend
    """
    if 'prop' in opts['lg'] and 'size' in opts['lg']['prop']:
        ax.tick_params(axis='both', which='major',
                       labelsize=opts['lg']['prop']['size'])

    leg = ['Theory', 'Simulation', 'Sim mid']
    if len(y) == 2:
        if ldata[1].shape[0] == 2:
            ln = ax.plot(x.ravel()[::sample], y[0].ravel()[::sample], 'g.',
                         x.ravel()[::sample], y[1].ravel()[::sample], 'b.')
            lt = ax.plot(ldata[0], ldata[1][0, :], 'r-',
                         ldata[0], ldata[1][1, :], 'r--', linewidth=2.0)
            leg2 = [x[:1] + ': ' + y for x in leg for y in titles[1:]]
            ax.legend(lt + ln, leg2, **opts['lg'], loc='lower right')
            ax.set_ylabel(titles[0], **opts['tx'])
        else:
            ln = ax.plot(x.ravel()[::sample], y[0].ravel()[::sample], 'g.',
                         x[::2, ::2].ravel()[::sample],
                         y[1].ravel()[::sample], 'b.')
            lt = ax.plot(ldata[0], ldata[1], 'r-', linewidth=2.0)
            leg2 = [x[:1] + ': ' + y for x in leg for y in titles[1:]]
            ax.legend(lt + ln, leg, **opts['lg'], loc='lower left')
    #        ax.set_title(titles[0] + ', '+ titles[1], **opts['tx'])
            ax.set_ylabel(titles[1], **opts['tx'])
    else:
        ln = ax.plot(x.ravel()[::sample], y[0].ravel()[::sample], 'g.')
        lt = ax.plot(ldata[0], ldata[1], 'r', linewidth=2.0)
        ax.legend(lt + ln, leg, **opts['lg'], loc='lower right')
#        ax.set_title(titles[0] + ', '+ titles[1], **opts['tx'])
        ax.set_ylabel(titles[1], **opts['tx'])
#        ax.set_title(titles[1], **opts['tx'])

    ax.set_xlabel(r'$\rho$', labelpad=-3, **opts['tx'])
    ax.set_xscale('log')
#    ax.plot(np.array([0, x.max()]), np.array([0, y[0].max()]), 'k-')
    ax.set_xlim((0., x.max()))
    ax.set_ylim((0., 1.1 * y[0].max()))
    ax.grid(b=True)
#    if len(y) ==2:
#        ax.plot(np.array([0, x.max()]), np.array([0, 0.5 * y[0].max()]), 'k-')
#    ax.set_ylabel('Simulation', **opts['tx'])


def make_hist(ax, thry, numl, num_bins, xlabl, titl, opts):  # Make histogram
    """
    Make histogram

    Parameters
    ----------
    ax
        axes object
    thry
        theoretical value
    numl
        list of ndarrays with simulation values
    num_bins
        number of bins
    xlabl
        list of strings [name, formula] of quantity
    titl
        string for title
    opts
        dictionary of options

        opts['tx']
            text options for x,y-axis label and title
        opts['lg']
            options for legend
    """
    if 'prop' in opts['lg'] and 'size' in opts['lg']['prop']:
        ax.tick_params(axis='both', which='major',
                       labelsize=opts['lg']['prop']['size'])

    ax.hist([numl[0].ravel(), numl[1].ravel()], bins=num_bins, normed=True)
    ax.set_xlim(left=0.0, right=numl[0].max())
    max_n = ax.get_ylim()[1]
    ln = ax.plot(np.array([thry, thry]), np.array([0, max_n]), 'r-')

#    ax.set_xlabel(xlabl[0] + ', ' + xlabl[1], **opts['tx'])
    ax.set_xlabel(xlabl[1], labelpad=-1, **opts['tx'])
    ax.set_ylabel('Relative frequency', **opts['tx'])
    ax.set_title(titl, **opts['tx'])
    ln[0].set_linewidth(2.0)
    ax.legend(ln, ['Theory'], **opts['lg'], loc='upper left')
    ln[0].set_data(np.array([[thry, thry], [0.0, ax.get_ylim()[1]]]))


def plot_data(ax, x, y, rho, cdata, ldata, titles, xylabl, cblab, opts, lpad,
              num_bins=10, sample=(1, 1)):  # plot data set
    """
    Plot data for one type of quantity

    Parameters
    ----------
    axh
        list of axes objects
    x,y
        ndarray of sigma^a positions
    rho
        ndarray of rho as a function of position
    ldata
        tuple of ndarrays, (x,y) of values for theory line on scatter plot
    cdata
        list of ndarrays of heatmap data
    titles
        list of strings for heatmap titles
    xylabl
        list of strings: [x-axis label, y-axis label]
    cblab
        list of strings for colorbar label [name, formula] of quantity
    opts
        dictionary of options

        opts['im']
            options for pcolormesh for heatmaps
        opts['asp']
            aspect ratio of pcolormesh for heatmaps
        opts['tx']
            text options for x,y-axis label and title
        opts['cb']
            text options for colorbar label
        opts['lg']
            options for legends
    lpad
        padding for colorbar label
    sample
        plot every sample'th point
    """
    make_heatmaps(ax[0:len(titles)], x, y, cdata[0:len(titles)],
                  xylabl, cblab, titles, opts, lpad, sample[0])
    if ldata is not None:
        make_scatter(ax[-1], rho, cdata[1:], ldata, cblab, opts,
                     10 * sample[1])
    else:
        make_hist(ax[-1], cdata[0][0, 0], cdata[1:], num_bins, cblab, '', opts)


# =============================================================================
# options
# =============================================================================


def default_options():
    """
    Default options for plotting data

    Returns
    -------
    opts = dict of dicts of options for plots

        opts['im']
            options for pcolormesh for heatmaps
        opts['asp']
            aspect ratio of pcolormesh for heatmaps
        opts['tx']
            text options for x,y-axis label and title
        opts['cb']
            text options for colorbar label
        opts['lg']
            options for legends
    labels
        dict of strings for titles, axis labels and legends

        labels['xy']
            list of strings: [x-axis label, y-axis label]
        labels['xyc']
            list of strings: [x-axis label, y-axis label]
        others:
                list of strings [name, formula(e)] of quantity(ies)
        labels['d']
            list of strings for distance
        labels['a']
            list of strings for angles
        labels['p']
            list of strings for projections
        labels['c']
            list of strings for curvature
    lpads
        tuple of padding lengths of colorbars for [distance, sine, curvature]
    samp
        plot every samp'th point, tuple of ints, (2,)
    """
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode'] = True
    mpl.rcParams['font.family'] = r'serif'

#    myim = {'origin':'lower', 'cmap':plt.get_cmap('viridis'),
#            'extent':(x[0], -x[0], y[0], -y[0])}
    imopts = {'cmap': plt.get_cmap('viridis')}
    imaspect = 1.  # 0.8 * intrinsic_range[1] / intrinsic_range[0]
    txtopts = {'size': 'xx-large', 'family': 'serif'}
    lgprops = {'prop': {'size': 'x-large', 'family': 'serif'}, 'numpoints': 1,
               'handlelength': 1, 'frameon': False, 'framealpha': 0.}
    cbtext = {'rotation': 270, 'labelpad': 27, **txtopts}
    lpads = (27, 20, 20, 20)
    sample = (4, 4)

    xylab = [r'$\Delta\sigma^1/\lambda^1$', r'$\Delta\sigma^2/\lambda^1$']
    xyclab = [r'$\sigma^1/\lambda^1$', r'$\sigma^2/\lambda^1$']
#    xylab = [r'Position difference, $\delta\sigma^1/\lambda_1$',
#             r'Position difference, $\delta\sigma^2/\lambda_1$']
#    xyclab = [r'Position, $\sigma^1/\lambda_1$',
#              r'Position, $\sigma^2/\lambda_1$']
    dlab = ['Euclidean distance',
            r'$\Vert\phi(\sigma)-\phi(\sigma^\prime)\Vert/\ell$']
    alab = [r'$\sin\theta_{a}$',
            r'$\sin\theta_{\mathrm{max}}$',
            r'$\sin\theta_{\mathrm{min}}$']
    plab = [r'$\cos\theta_{\mathcal{S}}$',
            r'$\cos\theta_{\mathcal{S}}$']
    clab = ['Curvature', r'$\mathcal{K}\ell^2$']

    opts = {'tx': txtopts, 'cb': cbtext, 'lg': lgprops, 'im': imopts,
            'asp': imaspect}
    labels = {'xy': xylab, 'xyc': xyclab, 'd': dlab, 'a': alab, 'p': plab,
              'c': clab}

    return opts, labels, lpads, sample


# =============================================================================
# running code
# =============================================================================


def load_and_plot(filename, opts, labels, lpads,
                  samp=(1, 1)):  # load data and plot
    """
    Load data from .npz file and plot

    Parameters
    ----------
    filenamee
        name of .npz file, w/o extension, for data
    opts = dict of dicts of options for plots

        opts['im']
            options for pcolormesh for heatmaps
        opts['asp']
            aspect ratio of pcolormesh for heatmaps
        opts['tx']
            text options for x,y-axis label and title
        opts['cb']
            text options for colorbar label
        opts['lg']
            options for legends
    labels
        dict of strings for titles, axis labels and legends

        labels['xy']
            list of strings: [x-axis label, y-axis label]
        labels['xyc']
            list of strings: [x-axis label, y-axis label]
        others:
                list of strings [name, formula(e)] of quantity(ies)
        labels['d']
            list of strings for distance
        labels['a']
            list of strings for angles
        labels['p']
            list of strings for projections
        labels['c']
            list of strings for curvature
    lpads
        tuple of padding lengths of colorbars for [distance, sine, curvature]
    samp
        plot every samp'th point, tuple of ints, (2,)
    """

    fig_d, ax_d = make_fig_ax((12.9, 3), 3)
    fig_a, ax_a = make_fig_ax((12.9, 3), 3)
    fig_p, ax_p = make_fig_ax((17.2, 3), 4)
    fig_c, ax_c = make_fig_ax((17.2, 3), 4)

    d = np.load(filename + '.npz')
    xc = np.append(d['x'], -d['x'][0])
    yc = np.append(d['y'], -d['y'][0])
    cdatad = [d['thr_dis'], d['num_dis']]
    ldatad = (d['rhol'], d['thr_disl'])
    cdataa = [d['thr_sin'][0], d['num_sin'][0], d['num_sin'][1]]
    ldataa = (d['rhol'], d['thr_sinl'])
    cdatap = [d['thr_pro'], d['num_pro'][0], d['num_pro'][1]]
    ldatap = (d['rhol'], d['thr_prol'])
    cdatac = [d['thr_cur'], d['num_cur'][0], d['num_cur'][1]]

    plot_data(ax_d, xc, yc, d['rho'], cdatad, ldatad, ['Theory', 'Simulation'],
              labels['xy'], labels['d'], opts, lpads[0], sample=samp)
    plot_data(ax_a, xc, yc, d['rho'], cdataa, ldataa, ['Theory', 'Simulation'],
              labels['xy'], labels['a'], opts, lpads[1], sample=samp)
    plot_data(ax_p, xc, yc, d['rho'], cdatap, ldatap,
              ['Theory', 'Simulation', 'Sim mid'],
              labels['xy'], labels['p'], opts, lpads[2], sample=samp)
    plot_data(ax_c, xc, yc, d['rho'], cdatac, None,
              ['Theory', 'Simulation 1', 'Simulation 2'],
              labels['xyc'], labels['c'], opts, lpads[3], sample=samp)

    d.close()

    return fig_d, fig_a, fig_p, fig_c


def load_plot_and_save(filename, opts, labels, lpads, fignames, figpath,
                       samp=(1, 1)):  # load data and plot
    """
    Load data from .npz file and plot

    Parameters
    ----------
    filenamee
        name of .npz file, w/o extension, for data
    opts = dict of dicts of options for plots

        opts['im']
            options for pcolormesh for heatmaps
        opts['asp']
            aspect ratio of pcolormesh for heatmaps
        opts['tx']
            text options for x,y-axis label and title
        opts['cb']
            text options for colorbar label
        opts['lg']
            options for legends
    labels
        dict of strings for titles, axis labels and legends

        labels['xy']
            list of strings: [x-axis label, y-axis label]
        labels['xyc']
            list of strings: [x-axis label, y-axis label]
        others:
                list of strings [name, formula(e)] of quantity(ies)
        labels['d']
            list of strings for distance
        labels['a']
            list of strings for angles
        labels['p']
            list of strings for projections
        labels['c']
            list of strings for curvature
    lpads
        tuple of padding lengths of colorbars for [distance, sine, curvature]
    fignames
        list of .pdf file names, w/o extensions or paths
    figpath
        path to folder for .pdf files, ending with '/'
    samp
        plot every samp'th point, tuple of ints, (2,)
    """

    figs = load_and_plot(filename, opts, labels, lpads, samp)

    for fig, figname in zip(figs, fignames):
        fig.savefig(figpath + figname + '.pdf', bbox_inches='tight')

# =============================================================================
# test code
# =============================================================================


if __name__ == "__main__":
    print('Run from outside package.')
