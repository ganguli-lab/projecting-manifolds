# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 21:37:24 2017

@author: subhy

Classes that provide tools for working with `numpy.linalg`'s broadcasting.

Classes
-------
array
    Subclass of `numpy.ndarray` with properties such as `inv` for matrix
    division, `t` for transposing stacks of matrices, `c`, `r` and `s` for
    dealing with stacks of vectors and scalars.
"""
from functools import wraps
import numpy as np
from numpy.lib.mixins import _numeric_methods
from ._gufuncs_cloop import pdist_ratio, cdist_ratio, norm  # matmul
from ._gufuncs_blas import matmul  # pdist_ratio, cdist_ratio, norm
from ._gufuncs_lapack import (tril_solve, rtriu_solve, qr, qr_c,
                              eigvalsh, singvals)
assert all((pdist_ratio, cdist_ratio, norm))
assert all((eigvalsh, singvals, qr, qr_c, tril_solve, rtriu_solve))
# =============================================================================
# Class: array
# =============================================================================


class array(np.ndarray):
    """Array object with linear algebra customisation.

    This is a subclass of `np.ndarray` with some added properties.
    The most important is matrix division via a lazy inverse.
    It also has some properties to work with broadcasting rules of `np.linalg`.

    Parameters
    ----------
    input_array : array_like
        The constructed array gets its data from `input_array`.
        Data is copied if necessary, as per `np.asarray`.

    Properties
    ----------
    t
        Transpose last two axes.
    c
        Insert singleton in last slot -> stack of column vectors.
    r
        Insert singleton in second last slot -> stack of row vectors.
    s
        Insert singleton in last two slots -> stack of scalars.
    uc, ur, us
        Undo effect of `r`, `c`, `s`.

    Examples
    --------
    >>> import numpy as np
    >>> import array as sp
    >>> x = sp.array(np.random.rand(2, 3, 4))
    >>> y = sp.array(np.random.rand(2, 3, 4))
    >>> u = x @ y.t
    >>> v = (x.r @ y.t).ur

    See also
    --------
    `np.ndarray` : the super class.
    `np.asarray` : used to get view/copy of data from `input_array`.
    """
    # Set of ufuncs that need special handling of vectors

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

#    def __array_finalize__(self, obj):
#        # We are not adding any attributes
#        pass

    __matmul__, __rmatmul__, __imatmul__ = _numeric_methods(matmul, 'matmul')

    @property
    def t(self) -> 'array':
        """Transpose last two indices.

        Transposing last two axes fits better with `np.linalg`'s
        broadcasting, which treats multi-dim arrays as stacks of matrices.

        Parameters/Results
        ------------------
        a : array, (..., M, N) --> transposed : array, (..., N, M)
        """
        return self.swapaxes(-1, -2)

    @property
    def r(self) -> 'array':
        """Treat multi-dim array as a stack of row vectors.

        Inserts a singleton axis in second-last slot.

        Parameters/Results
        ------------------
        a : array, (..., N) --> expanded : array, (..., 1, N)
        """
        return self[..., None, :]

    @property
    def c(self) -> 'array':
        """Treat multi-dim array as a stack of column vectors.

        Inserts a singleton axis in last slot.

        Parameters/Results
        ------------------
        a : array, (..., N) --> expanded : array, (..., N, 1)
        """
        return self[..., None]

    @property
    def s(self) -> 'array':
        """Treat multi-dim array as a stack of scalars.

        Inserts singleton axes in last two slots.

        Parameters/Results
        ------------------
        a : array, (...,) --> expanded : array, (..., 1, 1)
        """
        return self[..., None, None]

    @property
    def ur(self) -> 'array':
        """Undo effect of `r`.

        Parameters/Results
        ------------------
        a : array, (..., 1, N) --> squeezed : array, (..., N)

        Raises
        ------
        ValueError
            If a.shape[-2] != 1
        """
        return self.squeeze(axis=-2)

    @property
    def uc(self) -> 'array':
        """Undo effect of `c`.

        Parameters/Results
        ------------------
        a : array, (..., N, 1) --> squeezed : array, (..., N)

        Raises
        ------
        ValueError
            If a.shape[-1] != 1
        """
        return self.squeeze(axis=-1)

    @property
    def us(self) -> 'array':
        """Undo effect of `s`.

        Parameters/Results
        ------------------
        a : array, (..., 1, 1) --> squeezed : array, (...,)

        Raises
        ------
        ValueError
            If a.shape[-2] != 1 or a.shape[-1] != 1
        """
        return self.squeeze(axis=-2).squeeze(axis=-1)

    def flatter(self, start, stop) -> 'array':
        """Partial flattening.

        Flattens those axes in the range [start:stop)
        """
        newshape = self.shape[:start] + (-1,) + self.shape[stop:]
        return self.reshape(newshape)

    def expand_dims(self, axis) -> 'array':
        """Expand the shape of the array

        Alias of numpy.expand_dims.
        If `axis` is a sequence, axes are added one at a time, left to right.
        """
        if isinstance(axis, int):
            return np.expand_dims(self, axis).view(type(self))
        return self.expand_dims(axis[0]).expand_dims(axis[1:])


# =============================================================================
# Wrapping functionals
# =============================================================================


def wrap_one(np_func):
    """Create version of numpy function with single lnarray output.

    Does not pass through subclasses of `lnarray`

    Parameters
    ----------
    np_func : function
        A function that returns a single `ndarray`.

    Returns
    -------
    my_func : function
        A function that returns a single `lnarray`.
    """
    @wraps(np_func)
    def wrapped(*args, **kwargs):
        return np_func(*args, **kwargs).view(array)
    return wrapped
