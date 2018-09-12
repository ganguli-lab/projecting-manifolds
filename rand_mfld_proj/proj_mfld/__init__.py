# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:23:33 2017

@author: Subhy

Package: proj_mfld
=======================
Comparing simulations and formulae for distortion of random manifolds.

Modules
=======
rand_proj_mfld_num
    Simulate distortion of random manifolds under random projections.
rand_proj_mfld_theory
    Theoretical formulae for distortion of manifolds under random projections.
rand_proj_mfld_fit
    Linear fits to simulations.
rand_proj_mfld_plot
    Plot comparisons of simulations and theories.
"""
from . import rand_proj_mfld_num
from . import rand_proj_mfld_plot
from . import rand_proj_mfld_theory
from . import rand_proj_mfld_fit
from . import distratio
