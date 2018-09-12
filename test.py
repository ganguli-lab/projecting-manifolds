# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:27:45 2018

@author: Subhy
"""

from RandProjRandMan.MfldProj import distratio
import numpy as np

x = np.random.rand(163, 1000)
p = np.random.rand(163, 250)
dx, dn = distratio.pdist_ratio(x, p)
