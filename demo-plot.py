# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:16:58 2017

@author: Subhy

Quick demo of making plots
"""

from rand_mfld_proj import run


if __name__ == "__main__":
    """
    First block: quick demo of generating plots on demo data
    Second block: quick demo of generating plots on paper data
    """

    suffix = '_test'

#    suffix = ''

    run.icc_plot(False, suffix)
    run.ics_plot(False, suffix)
    run.gc_plot(False)
    run.gs_plot(False, suffix)
    run.rpm_plot(False, suffix)
    run.rpm_disp(suffix)
