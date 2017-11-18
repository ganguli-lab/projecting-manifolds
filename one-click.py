# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:14:08 2017

@author: Subhy

Generate data for plots in paper, save data, plot data and save plots.
"""

from RandProjRandMan import run


if __name__ == "__main__":
    """
    First block: generating & saving data for paper
    Second block: quick demo of generating & saving plots for paper
    """

    run.icc_data(True)
    run.ics_data(True)
    run.gs_data(True)
    run.rpm_num(True)

    run.icc_plot(True)
    run.ics_plot(True)
    run.gc_plot(True)
    run.gs_plot(True)
    run.rpm_plot(True)
    run.rpm_disp()
