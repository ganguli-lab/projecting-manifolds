# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:23:33 2017

@author: Subhy

Package: rand_mfld_proj
=======================
Code for paper: Lahiri, Gao, Ganguli, "Random projections of random manifolds".

Modules
=======
run
    Functions for simply generating data and plots.
disp_counter
    Displaying progress of `for` loops.

Subpackages
===========
proj
    Testing guarantee formulae for distortion.
mfld
    Testing formulae for geometry or random manifolds.
proj_mfld
    Comparing simulations and formulae for distortion of random manifolds.
"""
from . import proj
from . import mfld
from . import proj_mfld
from . import run
from . import iter_tricks