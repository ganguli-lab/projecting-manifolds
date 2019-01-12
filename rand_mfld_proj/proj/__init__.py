# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:23:33 2017

@author: Subhy

Package: proj
=======================
Testing guarantee formulae for distortion.

Modules
=======
inter_cell
    Simulate tests of distortion guarantee for long chords.
inter_cell_plot
    Plot tests of distortion guarantee for long chords.
intra_cell
    Simulate tests of distortion guarantee for tangent spaces.
intra_cell_plot
    Plot tests of distortion guarantee for tangent spaces.
"""

from . import inter_cell, inter_cell_plot, intra_cell, intra_cell_plot
assert all((inter_cell, inter_cell_plot, intra_cell, intra_cell_plot))
