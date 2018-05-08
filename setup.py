# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:22:44 2018

@author: Subhy
"""

from setuptools import setup, Extension, find_packages
# from distutils.core import setup, Extension
from numpy.lib import get_include
import os.path as osp

numpy_inc = get_include()
numpy_lib = osp.normpath(osp.join(numpy_inc, '..', 'lib'))

module1 = Extension('RandProjRandMan.MfldProj.distratio',
                    sources=['RandProjRandMan/MfldProj/distratio.c'],
                    include_dirs=[numpy_inc],
                    library_dirs=[numpy_lib],
                    libraries=['npymath'])
# extra_compile_args
# extra_link_args

setup(name='RandProjRandMan.MfldProj.distratio',
      version='1.0',
      description='Ratios pf cross/pair-wise distances squared',
      packages=find_packages(),
      ext_modules=[module1])
