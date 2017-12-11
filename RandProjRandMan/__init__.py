# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:23:33 2017

@author: Subhy

Package: RandProjRandMan
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
Cells
    Testing guarantee formulae for distortion.
RandCurve
    Testing formulae for geometry or random manifolds.
MfldProj
    Comparing simulations and formulae for distortion of random manifolds.
"""
from . import Cells
from . import RandCurve
from . import MfldProj
from . import run
from . import disp_counter