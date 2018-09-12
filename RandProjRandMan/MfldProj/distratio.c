/**
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 18:07:30 2018

@author: Subhy
"""
Adapted from https://github.com/scipy/scipy/scipy/spatial/src/distance_wrap.c
Copyright/licence info for that file:
 * Author: Damian Eads
 * Date:   September 22, 2007 (moved to new file on June 8, 2008)
 * Adapted for incorporation into Scipy, April 9, 2008.
 *
 * Copyright (c) 2007, Damian Eads. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   - Redistributions of source code must retain the above
 *     copyright notice, this list of conditions and the
 *     following disclaimer.
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer
 *     in the documentation and/or other materials provided with the
 *     distribution.
 *   - Neither the name of the author nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#if !defined(__clang__) && defined(__GNUC__) && defined(__GNUC_MINOR__)
#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 4)
/* enable auto-vectorizer */
#pragma GCC optimize("tree-vectorize")
/* float associativity required to vectorize reductions */
#pragma GCC optimize("unsafe-math-optimizations")
/* maybe 5% gain, manual unrolling with more accumulators would be better */
#pragma GCC optimize("unroll-loops")
#endif
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
/* Needs '.../numpy/core/include/' in the include path */
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

PyDoc_STRVAR(distratio__doc__,
"distratio\n=========\n"
"Maximum and minimum ratios of cross/pair-wise distances squared");

/* Calculations with C loops */
static NPY_INLINE double
euclidean_distance(const double *u, const double *v, const Py_ssize_t n)
{
    double s = 0.0;
    Py_ssize_t i;
    
    for (i = 0; i < n; ++i) {
        const double d = u[i] - v[i];
        s += d * d;
    }
    return s;
}

/* Doc string for Python function*/
PyDoc_STRVAR(cdist_ratio__doc__,
//"cdist_ratio(XA: ndarray, XB: ndarray, PA: ndarray, PB: ndarray) -> "
//"(drmax: float, drmin: float)\n\n"
"Maximum and minimum ratio of cross-wise distances squared beween corresponding "
"pairs of points in two groups of two sets.\n\n"
"Parameters\n-----------\n"
"XA: ndarray (P,N)\n"
"    Set of points *from* which we compute pairwise distances for the denominator. "
"    Each point is a row.\n"
"XB: ndarray (R,N)\n"
"    Set of points *to* which we compute pairwise distances for the denominator.\n"
"PA: ndarray (P,M)\n"
"    Set of points *from* which we compute pairwise distances for the numerator.\n"
"PB: ndarray (R,M)\n"
"    Set of points *to* which we compute pairwise distances for the numerator.\n"
"Returns\n-------\n"
"drmax: float\n"
"    Maximum ratio of distances squared.\n"
"drmin: float\n"
"    Minimum ratio of distances squared.\n");

/* Calculations with C loops */
static NPY_INLINE int
cdist_ratio_calc(const double *XA, const double *XB,
                 const double *PA, const double *PB,
                 double *drmax, double *drmin,
                 const Py_ssize_t num_rowsA, const Py_ssize_t num_rowsB,
                 const Py_ssize_t num_colsX, const Py_ssize_t num_colsP)
{
    Py_ssize_t i,j;
    *drmax = 0.0;
    *drmin = NPY_INFINITY;
    for (i = 0; i < num_rowsA; ++i) {
        const double *ux = XA + num_colsX * i;
        const double *up = PA + num_colsP * i;
        for (j = 0; j < num_rowsB; ++j) {
            const double *vx = XB + num_colsX * j;
            const double *vp = PB + num_colsP * j;
            const double ratio = euclidean_distance(up, vp, num_colsP)
                                 / euclidean_distance(ux, vx, num_colsX);
            if (ratio > *drmax)
                *drmax = ratio;
            if (ratio < *drmin)
                *drmin = ratio;
        }
    }
    return 0;
}

/* Wrapper: get data from Python, pass on to C loops */
static PyObject *
cdist_ratio_wrap(PyObject *self, PyObject *args)
{
    /* m-by-n matrices */
    PyArrayObject *XA_, *XB_, *PA_, *PB_;
    Py_ssize_t mA, mB, nX, nP;
    const double *XA, *XB, *PA, *PB;
    double drmax, drmin;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", 
                         &PyArray_Type, &XA_, &PyArray_Type, &XB_, 
                         &PyArray_Type, &PA_, &PyArray_Type, &PB_)) {
        return NULL;
    }
    else {
        NPY_BEGIN_ALLOW_THREADS;
        XA = PyArray_DATA(XA_);
        XB = PyArray_DATA(XB_);
        PA = PyArray_DATA(PA_);
        PB = PyArray_DATA(PB_);
        mA = PyArray_DIM(XA_, 0);
        mB = PyArray_DIM(XB_, 0);
        nX = PyArray_DIM(XA_, 1);
        nP = PyArray_DIM(PA_, 1);
        cdist_ratio_calc(XA, XB, PA, PB, &drmax, &drmin, mA, mB, nX, nP);
        NPY_END_ALLOW_THREADS;
    }
    return Py_BuildValue("dd", drmax, drmin);
}

/* Doc string for Python function*/
PyDoc_STRVAR(pdist_ratio__doc__,
//"pdist_ratio(X: ndarray, P: ndarray) -> (drmax: float, drmin: float)\n\n"
"Maximum and minimum ratio of pair-wise distances squared beween correspoinding "
"pairs of points in two sets.\n\n"
"Parameters\n-----------\n"
"X: ndarray (P,N)\n"
"    Set of points between which we compute pairwise distances for the denominator. "
"    Each point is a row.\n"
"P: ndarray (P,M)\n"
"    Set of points between which we compute pairwise distances for the numerator.\n"
"Returns\n-------\n"
"drmax: float\n"
"    Maximum ratio of distances squared.\n"
"drmin: float\n"
"    Minimum ratio of distances squared.\n");

/* Calculations with C loops */
static NPY_INLINE int
pdist_ratio_calc(const double *X, const double *P,
                 double *drmax, double *drmin,
                 const Py_ssize_t num_rows,
                 const Py_ssize_t num_colsX, const Py_ssize_t num_colsP)
{
    Py_ssize_t i,j;
    *drmax = 0.0;
    *drmin = NPY_INFINITY;
    for (i = 0; i < num_rows; ++i) {
        const double *ux = X + num_colsX * i;
        const double *up = P + num_colsP * i;
        for (j = i + 1; j < num_rows; ++j) {
            const double *vx = X + num_colsX * j;
            const double *vp = P + num_colsP * j;
            const double ratio = euclidean_distance(up, vp, num_colsP)
                                 / euclidean_distance(ux, vx, num_colsX);
            if (ratio > *drmax)
                *drmax = ratio;
            if (ratio < *drmin)
                *drmin = ratio;
        }
    }
    return 0;
}

/* Wrapper: get data from Python, pass on to C loops */
static PyObject *
pdist_ratio_wrap(PyObject *self, PyObject *args)
{
    /* m-by-n matrices */
    PyArrayObject *X_, *P_;
    Py_ssize_t m, nX, nP;
    const double *X, *P;
    double drmax, drmin;
    if (!PyArg_ParseTuple(args, "O!O!", 
                         &PyArray_Type, &X_, &PyArray_Type, &P_)) {
        return NULL;
    }
    else {
        NPY_BEGIN_ALLOW_THREADS;
        X = PyArray_DATA(X_);
        P = PyArray_DATA(P_);
        m = PyArray_DIM(X_, 0);
        nX = PyArray_DIM(X_, 1);
        nP = PyArray_DIM(P_, 1);
        pdist_ratio_calc(X, P, &drmax, &drmin, m, nX, nP);
        NPY_END_ALLOW_THREADS;
    }
    return Py_BuildValue("dd", drmax, drmin);
}

/* Module housekeeping */

static PyMethodDef distratioMethods[] = {
    {"cdist_ratio", cdist_ratio_wrap, METH_VARARGS, cdist_ratio__doc__},
    {"pdist_ratio", pdist_ratio_wrap, METH_VARARGS, pdist_ratio__doc__},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef distratiomod = {
    PyModuleDef_HEAD_INIT,
    "_distratio",      /* name of module */
    distratio__doc__, /* module documentation, may be NULL */
    -1,              /* size of per-interpreter state of the module,
                        or -1 if the module keeps state in global variables. */
    distratioMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit__distratio(void)
{
    PyObject *m;
    m = PyModule_Create(&distratiomod);
    if (m == NULL) {
        return;
    }
    import_array();  /* Must be present for NumPy.  Called first after above line.*/

    return m;
}