# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:22:44 2018

@author: Subhy
"""

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
from numpy.distutils.misc_util import get_info as get_misc_info
from distutils.sysconfig import get_python_inc
from numpy.distutils.system_info import get_info as get_sys_info

# =========================================================================
config = Configuration()

inc_dirs = [get_python_inc()]
if inc_dirs[0] != get_python_inc(plat_specific=1):
    inc_dirs.append(get_python_inc(plat_specific=1))
inc_dirs.append(get_numpy_include_dirs())

lapack_info = get_sys_info('lapack_opt', 0)  # and {}
npymath_info = get_misc_info("npymath")
all_info = {k: lapack_info[k] + npymath_info[k] for k in lapack_info.keys()}

# =============================================================================
config.add_extension('rand_mfld_proj._gufuncs_cloop',
                     sources=['rand_mfld_proj/gufuncs_cloop.c'],
                     include_dirs=inc_dirs,
                     extra_info=npymath_info)
# =============================================================================
config.add_extension('rand_mfld_proj._gufuncs_blas',
                     sources=['rand_mfld_proj/gufuncs_blas.c'],
                     include_dirs=inc_dirs,
                     extra_info=all_info)
# =============================================================================
config.add_extension('rand_mfld_proj._gufuncs_lapack',
                     sources=['rand_mfld_proj/gufuncs_lapack.c'],
                     include_dirs=inc_dirs,
                     extra_info=all_info)
# =============================================================================
setup(**config.todict())
# =============================================================================
