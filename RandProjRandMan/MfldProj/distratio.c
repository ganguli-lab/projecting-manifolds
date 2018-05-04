/**
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 18:07:30 2018

@author: Subhy
"""
*/

static NPY_INLINE double
euclidean_length(const double *u, const double *v, const npy_intp n)
{
    double s = 0.0;
    npy_intp i;
    
    for (i = 0; i < n; ++i) {
        const double d = u[i] - v[i];
        s += d * d;
    }
    return sqrt(s);
}

static NPY_INLINE int
cdist_ratio(const double *XA, const double *XB,
            const double *PA, const double *PB,
            double *drmax, double *drmin,
            const npy_intp num_rowsA, const npy_intp num_rowsB,
            const npy_intp num_colsX, const npy_intp num_colsP)
{
    Py_ssize_t i,j;
    *drmax = 0.0;
    *drmin = std::numeric_limits<double>::max();
    for (i = 0; i < num_rowsA; ++i) {
        const double *ux = XA + num_colsX * i;
        const double *up = PA + num_colsP * i;
        for (j = 0; j < num_rowsB; ++j) {
            const double *vx = XB + num_colsX * j;
            const double *vp = PB + num_colsP * j;
            const double ratio = euclidean_length(up, vp, num_colsP) / euclidean_length(ux, vx, num_colsX);
            if (ratio > *drmax)
                {*drmax = ratio;}
            if (ratio < *drmin)
                {*drmin = ratio;}
        }
    }
    return 0;
}

static NPY_INLINE int
pdist_ratio(const double *X,
            const double *P,
            double *drmax, double *drmin,
            const npy_intp num_rows,
            const npy_intp num_colsX, const npy_intp num_colsP)
{
    Py_ssize_t i,j;
    *drmax = 0.0;
    *drmin = std::numeric_limits<double>::max();
    for (i = 0; i < num_rows; ++i) {
        const double *ux = X + num_colsX * i;
        const double *up = P + num_colsP * i;
        for (j = 0; j < num_rows; ++j) {
            const double *vx = X + num_colsX * j;
            const double *vp = P + num_colsP * j;
            const double ratio = euclidean_length(up, vp, num_colsP) / euclidean_length(ux, vx, num_colsX);
            if (ratio > *drmax)
                {*drmax = ratio;}
            if (ratio < *drmin)
                {*drmin = ratio;}
        }
    }
    return 0;
}