# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:16:37 2017

@author: Subhy

Quick demo of generating data for plots
"""
from sl_py_tools.time_tricks import time_with
from rand_mfld_proj import run


if __name__ == "__main__":
    """
    First block: quick demo of generating & saving data
    """

#    run.icc_data(False, '_test')
#    run.ics_data(False, '_test')
#    run.gs_data(False, '_test')
#    run.rpm_num(False, '_test')

    with time_with():
        with time_with(absolute=False):
            run.icc_data(False, '_test')
        with time_with(absolute=False):
            run.ics_data(False, '_test')
        with time_with(absolute=False):
            run.gs_data(False, '_test')
        with time_with(absolute=False):
            run.rpm_num(False, '_test')
        print('Overall:')
