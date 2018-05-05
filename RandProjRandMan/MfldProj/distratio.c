/**
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 18:07:30 2018

@author: Subhy
"""
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
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

static NPY_INLINE double
euclidean_distance(const double *u, const double *v, const npy_intp n)
{
    double s = 0.0;
    npy_intp i;
    
    for (i = 0; i < n; ++i) {
        const double d = u[i] - v[i];
        s += d * d;
    }
    return npy_sqrt(s);
}

static NPY_INLINE int
cdist_ratio_calc(const double *XA, const double *XB,
                 const double *PA, const double *PB,
                 double *drmax, double *drmin,
                 const npy_intp num_rowsA, const npy_intp num_rowsB,
                 const npy_intp num_colsX, const npy_intp num_colsP)
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

static NPY_INLINE int
pdist_ratio_calc(const double *X,
                 const double *P,
                 double *drmax, double *drmin,
                 const npy_intp num_rows,
                 const npy_intp num_colsX, const npy_intp num_colsP)
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

static PyObject *
pdist_ratio_wrap(PyObject *self, PyObject *args)
{
    /* m-by-n matrices */
    PyArrayObject *X_, *P_;
    Py_ssize_t m, nX, nP;
    const double *X, *P;
    double drmax, drmin;
    if (!PyArg_ParseTuple(args, "O!O!O!O!", 
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

static PyMethodDef DistratioMethods[] = {
    {"cdist_ratio", cdist_ratio_wrap, METH_VARARGS, "ratio of cross-wise distances"},
    {"pdist_ratio", pdist_ratio_wrap, METH_VARARGS, "ratio of pair-wise distances"},
};

static struct PyModuleDef distratiomod = {
    PyModuleDef_HEAD_INIT,
    "distratio",   /* name of module */
    NULL,     /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    DistratioMethods
};

PyMODINIT_FUNC
PyInit_distratio(void)
{
    return PyModule_Create(&distratiomod);
}