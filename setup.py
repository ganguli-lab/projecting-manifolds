# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:22:44 2018

@author: Subhy
"""

from setuptools import setup, Extension
# from distutils.core import setup, Extension
from sysconfig import get_path
import os

numpy_dir = os.path.join(get_path('platlib'), 'numpy', 'core')

module1 = Extension('RandProjRandMan.MfldProj.distratio',
                    sources=['RandProjRandMan/MfldProj/distratio.c'],
                    include_dirs=[os.path.join(numpy_dir, 'include')])
#                    library_dirs=[os.path.join(numpy_dir, 'lib')],
#                    libraries=['npymath'])
# extra_compile_args
# extra_link_args

setup(name='RandProjRandMan.MfldProj.distratio',
      version='1.0',
      description='Ratios pf cross/pair-wise distances squared',
      ext_modules=[module1])
