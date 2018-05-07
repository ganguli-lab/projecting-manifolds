# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:22:44 2018

@author: Subhy
"""

from distutils.core import setup, Extension

module1 = Extension('MfldProj.distratio',
                    sources=['MfldProj/distratio.c'],
                    include_dirs=[r'C:\Anaconda3\Lib\site-packages\numpy\core\include'],
                    library_dirs=[r'C:\Anaconda3\Lib\site-packages\numpy\core\lib'],
                    libraries=['npymath'])

setup(name='MfldProj.distratio',
      version='1.0',
      description='Ratios of cross/pair-wise distances',
      ext_modules=[module1])
