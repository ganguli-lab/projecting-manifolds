# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:00:23 2017

@author: Subhy

Module: run
===========
Functions for simply running code to generate or plopt data.

Functions
=========
icc_data
    Generate data for Figure 2, chordal cone guarantee
icc_plot
    Plot data for Figure 2, chordal cone guarantee
ics_data
    Generate data for Figure 3, tangential cone guarantee
ics_plot
    Plot data for Figure 3, tangential cone guarantee
gc_plot
    Generate and plot data for Figure 4, random curve geometry
gs_data
    Generate data for Figure 5, random surface geometry
gs_plot
    Plot data for Figure 5, random surface geometry
rpm_num
    Generate data for Figure Figures 6&9, distortion of random manifolds
rpm_plot
    Plot data for Figure Figures 6&9, distortion of random manifolds
rpm_disp
    Display linear fits for Figure Figures 6&9, distortion of random manifolds
"""
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
from .MfldProj import rand_proj_mfld_num as rpmn
from .MfldProj import rand_proj_mfld_plot as rpmp
from .RandCurve import gauss_curve as gc
from .RandCurve import gauss_curve_plot as gcp
from .RandCurve import gauss_surf as gs
from .RandCurve import gauss_surf_plot as gsp
from .Cells import inter_cell as icc
from .Cells import inter_cell_plot as iccp
from .Cells import intra_cell as ics
from .Cells import intra_cell_plot as icsp

data_dir = 'Data/'
fig_dir = 'Figures/'

# =============================================================================
# Cell distortion
# =============================================================================


def icc_data(long: bool=False, suffix: str=''):
    """
    Generate data for Figure 2, chordal cone guarantee, and save in a .npz file

    Parameters
    ==========
    long : bool = False
        If true, use parameters for a long simulationnused in the paper,
        otherwise use parameters for a quick demo.
    suffix : str = ''
        appended to name of .npz file.
    """
#    # choose parameters
#    np.random.seed(0)
#    # number of samples of edge of cone
#    num_trials = 2000000
#    # number of times to repeat each comparison
#    num_reps = 5
#    # dimensionality of ambient space
#    ambient_dim = 1000
#    # dimensionality of projection
#    proj_dims = [50, 75, 100]
#    # angle between cone centre and edge
#    thetas = [0.001, 0.002, 0.003, 0.004]
    if long:
        opts = icc.default_options()
    else:
        opts = icc.quick_options()

    icc.make_and_save(data_dir + 'intercell' + suffix, *opts)


def icc_plot(save: bool=False, suffix: str=''):
    """
    Load data from .npz file for Figure 2, chordal cone guarantee, make plots,
    and save as .pdf files if requested.

    Parameters
    ==========
    save : bool = False
        If true, save figures as .pdf files, otherwise just display.
    suffix : str = ''
        appended to name of .npz file.
    """
    pmrks, pcols, txtopts, legopts, siz = iccp.default_options()
    data_loc = data_dir + 'intercell' + suffix

    if save:
        iccp.load_and_plot_and_save(data_loc, pmrks, pcols,
                                    fig_dir + 'inter_cell',
                                    txtopts, legopts, siz)
    else:
        iccp.load_and_plot(data_loc, pmrks, pcols,
                           txtopts, legopts, siz)

    plt.show()


def ics_data(long: bool=False, suffix: str=''):
    """
    Generate data for Figure 3, tangential cone guarantee, and save in a .npz
    file

    Parameters
    ==========
    long : bool = False
        If true, use parameters for a long simulationnused in the paper,
        otherwise use parameters for a quick demo.
    suffix : str = ''
        appended to name of .npz file.
    """
#    # choose parameters
#    np.random.seed(0)
#    # number of samples of edge of cone
#    num_trials = 200000
#    # number of times to repeat each comparison
#    num_reps = 5
#    # dimensionality of ambient space
#    ambient_dim = 1000
#    # dimensionality of projection
#    proj_dims = [50, 75, 100]
#    # dimensionality of subspace
#    sub_dims = [5, 10]
#    # max angle between cone centre and edge
#    thetas = [0.001, 0.002, 0.003, 0.004]
    if long:
        opts = ics.default_options()
    else:
        opts = ics.quick_options()

    ics.make_and_save(data_dir + 'intracell_trial' + suffix, 1000, *opts[1:])


def ics_plot(save: bool=False, suffix: str=''):
    """
    Load data from .npz file for Figure 3, tangential cone guarantee, make
    plots, and save as .pdf files if requested.

    Parameters
    ==========
    save : bool = False
        If true, save figures as .pdf files, otherwise just display.
    suffix : str = ''
        appended to name of .npz file.
    """
    pmrks, pcols, pfills, txtopts, legopts, siz = icsp.default_options()
    data_loc = data_dir + 'intracell' + suffix

    if save:
        icsp.load_and_plot_and_save(data_loc, pmrks, pcols,
                                    pfills, fig_dir + 'intra_cell',
                                    txtopts, legopts, siz)
    else:
        icsp.load_and_plot(data_loc, pmrks, pcols, pfills,
                           txtopts, legopts, siz)

    plt.show()


# =============================================================================
# Manifold geometry
# =============================================================================


def gc_plot(save: bool=False):
    """
    Generate data for Figure 4, random curve geometry, make plots, and save as
    .pdf files if requested.

    test

    Parameters
    ==========
    save : bool = False
        If true, save figures as .pdf files, otherwise just display.
    """
#    np.random.seed(0)
#    ambient_dim = 1000     # dimensionality of ambient space
#    intrinsic_range = 5.0  # x-coordinate lies between +/- this
#    intrinsic_num = 1024   # number of points to sample
#    num_trials = 5         # number of additional numerical comparisons

    (num_trials, ambient_dim, intrinsic_range,
     intrinsic_num) = gc.default_options_data()
    xlabs, ylabs, txtopts, legopts, leglocs = gcp.default_options_plot()

    figs = gc.make_and_plot(num_trials, ambient_dim, intrinsic_range,
                            intrinsic_num, xlabs, ylabs, txtopts, legopts,
                            leglocs)

    if save:
        gcp.save_figs_all(figs, ['distance_1d', 'angle_1d', 'proj_1d',
                                 'curvature_1d'], fig_dir)

    plt.show()


def gs_data(long: bool=False, suffix: str=''):
    """
    Generate data for Figure 5, random surface geometry, and save in a .npz
    file.

    Parameters
    ==========
    long : bool = False
        If true, use parameters for a long simulationnused in the paper,
        otherwise use parameters for a quick demo.
    suffix : str = ''
        appended to name of .npz file.
    """
    data_loc = data_dir + 'randsurf' + suffix
    if long:
        opts = gs.default_options()
#        ambient_dim = 200    # dimensionality of ambient space
#        intrinsic_range = (6.0, 10.0)  # x-coordinate lies between +/- this
#        intrinsic_num = (128, 256)  # number of points to sample
#        width = (1.0, 1.8)
        gs.make_and_save(data_loc, *opts)
    else:
        # choose parameters
        opts = gs.quick_options()
        gs.make_and_save(data_loc, *opts)


def gs_plot(save: bool=False, suffix: str=''):
    """
    Load data from .npz file for Figure 5, random surface geometry, make plots,
    and save as .pdf files if requested.

    Parameters
    ==========
    save : bool = False
        If true, save figures as .pdf files, otherwise just display.
    suffix : str = ''
        appended to name of .npz file.
    """
    myopts, mylabs, lpads, sample = gsp.default_options()

    fignms = ['distance_2d', 'angle_2d', 'proj_2d', 'curvature_2d']
    dataloc = data_dir + 'randsurf' + suffix

    if save:
        gsp.load_plot_and_save(dataloc, myopts, mylabs, lpads, fignms, fig_dir,
                               samp=sample)
    else:
        gsp.load_and_plot(dataloc, myopts, mylabs, lpads, samp=sample)

    plt.show()


# =============================================================================
# Manifold distortion
# =============================================================================


def rpm_num(long: bool=False, suffix: str=''):
    """
    Generate data for Figures 6&9, distortion of random manifolds, and
    save in a .npz file.

    Parameters
    ==========
    long : bool = False
        If true, use parameters for a long simulationnused in the paper,
        otherwise use parameters for a quick demo.
    suffix : str = ''
        appended to name of .npz file.
    """
    np.random.seed(0)
    data_loc = data_dir + 'randmanproj' + suffix
    if long:
        opts = rpmn.default_options()
#        epsilons = [0.2, 0.3, 0.4]
#        proj_dims = (np.linspace(5, 250, 50, dtype=int).tolist(),
#                     np.linspace(5, 250, 50, dtype=int).tolist())
#        # dimensionality of ambient space
#        ambient_dims = ((250 * np.logspace(0, 6, num=7, base=2,
#                                           dtype=int)).tolist(), 1000)
#        mfld_fracs = np.logspace(-1.5, 0, num=10, base=5).tolist()
#        prob = 0.05
#        num_samp = 100
#        # number of points to sample
#        intrinsic_num = ((40, 40), (100, 100))
#        # x-coordinate lies between +/- this
#        intrinsic_range = ((20.0, 20.0), (50.0, 50.0))
#        width = (8.0, 8.0)
        #
        rpmn.make_and_save(data_loc, *opts)
        #
    else:
        opts = rpmn.quick_options()
        rpmn.make_and_save(data_loc, *opts)


def rpm_plot(save: bool=False, suffix: str=''):
    """
    Load data from .npz file for Figures 6&9, distortion of random manifolds,
    make plots, and save as .pdf files if requested.

    Parameters
    ==========
    save : bool = False
        If true, save figures as .pdf files, otherwise just display.
    suffix : str = ''
        appended to name of .npz file.
    """
    data_loc = data_dir + 'randmanproj' + suffix
    plot_vars = ['N', 'V']
    plot_froms = ['all', 'sim', 'LGG', 'BW', 'Vr', 'EW']
    fignms = []
    for plot_from in plot_froms:
        for plot_var in plot_vars:
            fignms.append('M_' + plot_var + '_' + plot_from)

    kwargs = {'linewidth': 2.}
    opts, labels, styleK, styleE, styleF = rpmp.default_options()

    if save:
        rpmp.load_and_plot_and_save(data_loc, opts, labels, fignms, fig_dir,
                                    styleK, styleE, styleF, **kwargs)
    else:
        rpmp.load_and_plot(data_loc, opts, labels, styleK, styleE, styleF,
                           **kwargs)
    plt.show()


def rpm_disp(suffix: str=''):
    """
    Load data from .npz file for Figures 6&9, distortion of random manifolds,
    and display linear fits.

    Parameters
    ==========
    suffix : str = ''
        appended to name of .npz file.
    """

    data_loc = data_dir + 'randmanproj' + suffix

    rpmp.load_and_fit(data_loc, np.array([0, 3, 4]))


# =============================================================================
# Running code
# =============================================================================


if __name__ == "__main__":
    """If file is run"""
    print('Run from outside package.')
