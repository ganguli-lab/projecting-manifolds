/**
# -*- c -*-
# -*- coding: utf-8 -*-
*/
/*
Adapted from https://github.com/numpy/numpy/numpy/linalg/umath_linalg.c.src
Copyright/licence info for that file:
 * Copyright (c) 2005-2017, NumPy Developers.
 * All rights reserved.
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

/*              Table of Contents
56.   Includes
75.   Docstrings
146.  Outer loop macros
220.  BLAS/Lapack calling functions
315.  Error signaling functions
340.  Constants
381.  Structs used for data rearrangement
457.  Data rearrangement functions
708.  QR
946.  SOLVE
1087. LSTSQ
1281. EIGVALS
1447. SINGVALS
1130. Ufunc definition
1198. Module initialization stuff
*/

/*
 *****************************************************************************
 **                            INCLUDES                                     **
 *****************************************************************************
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "Python.h"
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_math.h"

#include "numpy/npy_3kcompat.h"

// #include "npy_config.h"

static const char* gufuncs_lapack_version_string = "0.1.0";

/*
 *****************************************************************************
 **                   Doc string for Python functions                       **
 *****************************************************************************
 */

PyDoc_STRVAR(qr__doc__,
//"qr(X: ndarray, Y: ndarray) -> (Z: ndarray)\n\n"
"QR decomposition.\n\n"
"Factor a matrix as `A = QR` with `Q` orthogonal and `R` upper-triangular.\n"
"`K` = `M` or `N`, depending on which of `qr_m` or `qr_n` was called. "
"When M < N, `qr_n` cannot be called.\n\n"
"Parameters\n-----------\n"
"A: ndarray (M,N)\n"
"    Matrix to be factored.\n\n"
"Returns\n-------\n"
"Q: ndarray (M,K)\n"
"    Matrix with orthonormal columns.\n"
"R: ndarray (K,N)\n"
"    Matrix with zeros below the diagonal.");

PyDoc_STRVAR(solve__doc__,
//"solve(A: ndarray, B: ndarray) -> (C: ndarray)\n\n"
"Solve linear system.\n\n"
"Solve the equation `AX = B` for `X`.\n\n"
"Parameters\n-----------\n"
"A: ndarray (N,N)\n"
"    Matrix of coefficients.\n"
"B: ndarray (N,NRHS)\n"
"    Matrix of result vectors.\n\n"
"Returns\n-------\n"
"X: ndarray (N,NRHS)\n"
"    Matrix of solution vectors.\n");

PyDoc_STRVAR(lstsq__doc__,
//"lstsq(A: ndarray, B: ndarray) -> (C: ndarray)\n\n"
"Least-square solution of linear system.\n\n"
"Find the least-square solution of the equation `AX = B` for `X`.\n\n"
"Parameters\n-----------\n"
"A: ndarray (M,N)\n"
"    Matrix of coefficients.\n"
"B: ndarray (M,NRHS)\n"
"    Matrix of result vectors.\n\n"
"Returns\n-------\n"
"X: ndarray (N,NRHS)\n"
"    Matrix of solution vectors.\n");

PyDoc_STRVAR(eigvalsh__doc__,
//"eigvalsh(A: ndarray) -> (L: ndarray)\n\n"
"Eigenvalues of hermitian matrix.\n\n"
"Find the `lambda` such that `A x == lambda x` for some `x`.\n\n"
"Parameters\n-----------\n"
"A: ndarray (...,N,N)\n"
"    Matrix of coefficients.\n"
"Returns\n-------\n"
"L: ndarray (...,N)\n"
"    Vector of eigenvalues.\n");

PyDoc_STRVAR(singvals__doc__,
//"eigvals(A: ndarray) -> (S: ndarray)\n\n"
"Singular values of matrix.\n\n"
"Find the `s` such that `A v == s u, A' u == s v,` for some `u,v`.\n\n"
"Parameters\n-----------\n"
"A: ndarray (...,N,N)\n"
"    Matrix of coefficients.\n"
"Returns\n-------\n"
"S: ndarray (...,K)\n"
"    Vector of singular values. `K = min(M,N)`\n");

/*
 *****************************************************************************
 **                         OUTER LOOP MACROS                               **
 *****************************************************************************
*/

typedef int               fortran_int;
typedef float             fortran_real;
typedef double            fortran_doublereal;

static NPY_INLINE fortran_int
fortran_int_min(fortran_int x, fortran_int y) {
 return x < y ? x : y;
}

static NPY_INLINE fortran_int
fortran_int_max(fortran_int x, fortran_int y) {
 return x > y ? x : y;
}

#define INIT_OUTER_LOOP_1       \
 npy_intp dN = *dimensions++;\
 npy_intp N_;                \
 npy_intp s0 = *steps++;

#define INIT_OUTER_LOOP_2       \
 INIT_OUTER_LOOP_1           \
 npy_intp s1 = *steps++;

#define INIT_OUTER_LOOP_3       \
 INIT_OUTER_LOOP_2           \
 npy_intp s2 = *steps++;

#define INIT_OUTER_LOOP_5 \
 INIT_OUTER_LOOP_4\
 npy_intp s4 = *steps++;

#define INIT_OUTER_LOOP_6  \
 INIT_OUTER_LOOP_5\
 npy_intp s5 = *steps++;

#define INIT_OUTER_LOOP_4       \
 INIT_OUTER_LOOP_3           \
 npy_intp s3 = *steps++;

#define BEGIN_OUTER_LOOP_2      \
 for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1) {

#define BEGIN_OUTER_LOOP_3      \
 for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2) {

#define BEGIN_OUTER_LOOP_4      \
 for (N_ = 0; N_ < dN; N_++, args[0] += s0, args[1] += s1, args[2] += s2, args[3] += s3) {

#define BEGIN_OUTER_LOOP_5 \
 for (N_ = 0;\
      N_ < dN;\
      N_++, args[0] += s0,\
          args[1] += s1,\
          args[2] += s2,\
          args[3] += s3,\
          args[4] += s4) {

#define BEGIN_OUTER_LOOP_6 \
 for (N_ = 0;\
      N_ < dN;\
      N_++, args[0] += s0,\
          args[1] += s1,\
          args[2] += s2,\
          args[3] += s3,\
          args[4] += s4,\
          args[5] += s5) {

#define END_OUTER_LOOP  }
/*
*****************************************************************************
*                    BLAS/LAPACK calling macros                             *
*****************************************************************************
*/

#ifdef NO_APPEND_FORTRAN
# define FNAME(x) x
#else
# define FNAME(x) x##_
#endif

/* copy vector x into y */
extern int
FNAME(scopy)(int *n,
             float *sx, int *incx,
             float *sy, int *incy);
extern int
FNAME(dcopy)(int *n,
             double *sx, int *incx,
             double *sy, int *incy);

/* qr decomposition of a */
/* a -> r, v, tau */
extern int
FNAME(sgeqrf)(int *m, int *n, float *a, int *lda, float *tau,
              float *work, int * lwork, int *info);

extern int
FNAME(dgeqrf)(int *m, int *n, double *a, int *lda, double *tau,
              double *work, int * lwork, int *info);

/* v, tau -> q */
extern int
FNAME(sorgqr)(int *m, int *n, int *k,
             float *a, int *lda, float *tau,
             float *work, int * lwork, int *info);

extern int
FNAME(dorgqr)(int *m, int *n, int *k,
             double *a, int *lda, double *tau,
             double *work, int * lwork, int *info);;

/* solve a x = b for x */
extern int
FNAME(sgesv)(int *n, int *nrhs,
            float *a, int *lda, int * ipiv,
            float *b, int *ldb, int *info);

extern int
FNAME(dgesv)(int *n, int *nrhs,
             double *a, int *lda, int * ipiv,
             double *b, int *ldb, int *info);

/* least square solution of a x = b for x */
extern int
FNAME(sgelsd)(int *m, int *n, int *nrhs,
             float *a, int *lda, float *b, int *ldb,
             float *s, float *rcond, int *rank,
             float *work, int *lwork, int *iwork, int *info);

extern int
FNAME(dgelsd)(int *m, int *n, int *nrhs,
             double *a, int *lda, double *b, int *ldb,
             double *s, double *rcond, int *rank,
             double *work, int *lwork, int *iwork, int *info);

/* eigenvalue decomposition */
extern int
FNAME(ssyevd)(char *jobz, char *uplo, int *n,
             float *a, int *lda, float *w,
             float *work, int *lwork, int *iwork, int *liwork, int *info);

extern int
FNAME(dsyevd)(char *jobz, char *uplo, int *n,
             double *a, int *lda, double *w,
             double *work, int *lwork, int *iwork, int *liwork, int *info);

/* singular value decomposition */
extern int
FNAME(sgesdd)(char *jobz, int *m, int *n,
             float *a, int *lda, float *s, float *u, int *ldu, float *v, int *ldv,
             float *work, int *lwork, int *iwork, int *info);

extern int
FNAME(dgesdd)(char *jobz, int *m, int *n,
             double *a, int *lda, double *s, double *u, int *ldu, double *v, int *ldv,
             double *work, int *lwork, int *iwork, int *info);

#define BLAS(FUNC)                              \
    FNAME(FUNC)

#define LAPACK(FUNC)                            \
    FNAME(FUNC)

/*
 *****************************************************************************
 **                      Error signaling functions                          **
 *****************************************************************************
 */

static NPY_INLINE int
get_fp_invalid_and_clear(void)
{
    int status;
    status = npy_clear_floatstatus_barrier((char*)&status);
    return !!(status & NPY_FPE_INVALID);
}

static NPY_INLINE void
set_fp_invalid_or_clear(int error_occurred)
{
    if (error_occurred) {
        npy_set_floatstatus_invalid();
    }
    else {
        npy_clear_floatstatus_barrier((char*)&error_occurred);
    }
}

/*
 *****************************************************************************
 **                      Some handy constants                               **
 *****************************************************************************
 */

 static float s_one;
 static float s_zero;
 static float s_minus_one;
 static float s_inf;
 static float s_nan;
 static float s_eps;
 static double d_one;
 static double d_zero;
 static double d_minus_one;
 static double d_inf;
 static double d_nan;
 static double d_eps;

static void init_constants(void)
{
    /*
    this is needed as NPY_INFINITY and NPY_NAN macros
    can't be used as initializers. I prefer to just set
    all the constants the same way.
    */
    s_one  = 1.0f;
    s_zero = 0.0f;
    s_minus_one = -1.0f;
    s_inf = NPY_INFINITYF;
    s_nan = NPY_NANF;
    s_eps = npy_spacingf(s_one);

    d_one  = 1.0;
    d_zero = 0.0;
    d_minus_one = -1.0;
    d_inf = NPY_INFINITY;
    d_nan = NPY_NAN;
    d_eps = npy_spacing(d_one);
}

/*
 *****************************************************************************
 **               Structs used for data rearrangement                       **
 *****************************************************************************
 */

/*
 * this struct contains information about how to linearize a matrix in a local
 * buffer so that it can be used by blas functions.  All strides are specified
 * in bytes and are converted to elements later in type specific functions.
 *
 * rows: number of rows in the matrix
 * columns: number of columns in the matrix
 * row_strides: the number bytes between consecutive rows.
 * column_strides: the number of bytes between consecutive columns.
 * output_lead_dim: BLAS/LAPACK-side leading dimension, in elements
 */
typedef struct linearize_data_struct
{
    npy_intp rows;
    npy_intp columns;
    npy_intp row_strides;
    npy_intp column_strides;
    npy_intp output_lead_dim;
} LINEARIZE_DATA_t;

static NPY_INLINE void
init_linearize_data_ex(LINEARIZE_DATA_t *lin_data,
                        npy_intp rows,
                        npy_intp columns,
                        npy_intp row_strides,
                        npy_intp column_strides,
                        npy_intp output_lead_dim)
{
    lin_data->rows = rows;
    lin_data->columns = columns;
    lin_data->row_strides = row_strides;
    lin_data->column_strides = column_strides;
    lin_data->output_lead_dim = output_lead_dim;
}

static NPY_INLINE void
init_linearize_data(LINEARIZE_DATA_t *lin_data,
                     npy_intp rows,
                     npy_intp columns,
                     npy_intp row_strides,
                     npy_intp column_strides)
{
    init_linearize_data_ex(
        lin_data, rows, columns, row_strides, column_strides, columns);
}

/*
 * this struct contains information about how to linearize a vector in a local
 * buffer so that it can be used by blas functions.  All strides are specified
 * in bytes and are converted to elements later in type specific functions.
 *
 * len: number of elements in the vector
 * strides: the number bytes between consecutive elements.
 */
typedef struct linearize_vdata_struct
{
  npy_intp len;
  npy_intp strides;
} LINEARIZE_VDATA_t;


static NPY_INLINE void
init_linearize_vdata(LINEARIZE_VDATA_t *lin_data,
                    npy_intp len,
                    npy_intp strides)
{
    lin_data->len = len;
    lin_data->strides = strides;
}

/*
 *****************************************************************************
 **                    DATA REARRANGEMENT FUNCTIONS                         **
 *****************************************************************************
 */

              /* rearranging of 2D matrices using blas */

#line 471

static NPY_INLINE void *
linearize_FLOAT_matrix(void *dst_in,
                         const void *src_in,
                         const LINEARIZE_DATA_t* data)
 {
    float *src = (float *) src_in;
    float *dst = (float *) dst_in;

    if (dst) {
        int i, j;
        float* rv = dst;
        fortran_int columns = (fortran_int)data->columns;
        fortran_int column_strides =
                (fortran_int)(data->column_strides/sizeof(float));
        fortran_int one = 1;
        for (i = 0; i < data->rows; i++) {
            if (column_strides > 0) {
                FNAME(scopy)(&columns,
                              (void*)src, &column_strides,
                              (void*)dst, &one);
            }
            else if (column_strides < 0) {
                FNAME(scopy)(&columns,
                              (void*)((float*)src + (columns-1)*column_strides),
                              &column_strides,
                              (void*)dst, &one);
            }
            else {
            /*
             * Zero stride has undefined behavior in some BLAS
             * implementations (e.g. OSX Accelerate), so do it
             * manually
             */
            for (j = 0; j < columns; ++j) {
                memcpy((float*)dst + j, (float*)src, sizeof(float));
            }
            }
            src += data->row_strides/sizeof(float);
            dst += data->output_lead_dim;
        }
        return rv;
    } else {
        return src;
    }
}

static NPY_INLINE void *
delinearize_FLOAT_matrix(void *dst_in,
                        const void *src_in,
                        const LINEARIZE_DATA_t* data)
{
    float *src = (float *) src_in;
    float *dst = (float *) dst_in;

    if (src) {
        int i;
        float *rv = src;
        fortran_int columns = (fortran_int)data->columns;
        fortran_int column_strides =
          (fortran_int)(data->column_strides/sizeof(float));
        fortran_int one = 1;
        for (i = 0; i < data->rows; i++) {
            if (column_strides > 0) {
                FNAME(scopy)(&columns,
                              (void*)src, &one,
                              (void*)dst, &column_strides);
            }
            else if (column_strides < 0) {
                FNAME(scopy)(&columns,
                              (void*)src, &one,
                              (void*)((float*)dst + (columns-1)*column_strides),
                              &column_strides);
            }
            else {
              /*
               * Zero stride has undefined behavior in some BLAS
               * implementations (e.g. OSX Accelerate), so do it
               * manually
               */
                if (columns > 0) {
                    memcpy((float*)dst,
                           (float*)src + (columns-1),
                           sizeof(float));
              }
            }
            src += data->output_lead_dim;
            dst += data->row_strides/sizeof(float);
        }
        return rv;
    } else {
        return src;
    }
}

static NPY_INLINE void *
delinearize_FLOAT_triu(void *dst_in,
                        const void *src_in,
                        const LINEARIZE_DATA_t* data)
{
   float *src = (float *) src_in;
   float *dst = (float *) dst_in;

   if (src) {
       int i;
        float *rv = src;
        fortran_int columns = (fortran_int)data->columns;
        fortran_int column_strides =
            (fortran_int)(data->column_strides/sizeof(float));
        fortran_int one = 1;
        for (i = 0; i < data->rows; i++) {
            fortran_int n = fortran_int_min(i + one, columns);
            if (column_strides > 0) {
                FNAME(scopy)(&n,
                              (void*)src, &one,
                              (void*)dst, &column_strides);
            }
            else if (column_strides < 0) {
                FNAME(scopy)(&n,
                              (void*)src, &one,
                              (void*)((float*)dst + (n-1)*column_strides),
                              &column_strides);
            }
            else {
               /*
                * Zero stride has undefined behavior in some BLAS
                * implementations (e.g. OSX Accelerate), so do it
                * manually
                */
                if (columns > 0) {
                    memcpy((float*)dst,
                           (float*)src + (columns-1),
                           sizeof(float));
                }
            }
            src += data->output_lead_dim;
            dst += data->row_strides/sizeof(float);
        }

        return rv;
    } else {
        return src;
    }
}

static NPY_INLINE void
nan_FLOAT_matrix(void *dst_in, const LINEARIZE_DATA_t* data)
{
    float *dst = (float *) dst_in;

    int i, j;
    ptrdiff_t cs = data->column_strides/sizeof(float);
    for (i = 0; i < data->rows; i++) {
        float *cp = dst;
        for (j = 0; j < data->columns; ++j) {
            *cp = s_nan;
            cp += cs;
        }
        dst += data->row_strides/sizeof(float);
    }
}

static NPY_INLINE void
zero_FLOAT_matrix(void *dst_in, const LINEARIZE_DATA_t* data)
{
    float *dst = (float *) dst_in;

    int i, j;
    ptrdiff_t cs = data->column_strides/sizeof(float);
    for (i = 0; i < data->rows; i++) {
        float *cp = dst;
        for (j = 0; j < data->columns; ++j) {
            *cp = s_zero;
            cp += cs;
        }
        dst += data->row_strides/sizeof(float);
    }
}

static NPY_INLINE void *
delinearize_FLOAT_vec(void *dst_in,
                     void *src_in,
                     const LINEARIZE_VDATA_t *data)
{
    float *src = (float *) src_in;
    float *dst = (float *) dst_in;

    if (dst) {
        float* rv = dst;
        fortran_int len = (fortran_int)data->len;
        fortran_int strides = (fortran_int)(data->strides/sizeof(float));
        fortran_int one = 1;
        if (strides > 0) {
            FNAME(scopy)(&len,
                          (void*)src, &one,
                          (void*)dst, &strides);
        }
        else if (strides < 0) {
            FNAME(scopy)(&len,
                          (void*)((float*)src + (len-1)*strides),
                          &one,
                          (void*)dst, &strides);
        }
        else {
            /*
             * Zero stride has undefined behavior in some BLAS
             * implementations (e.g. OSX Accelerate), so do it
             * manually
             */
            int j;
            for (j = 0; j < len; ++j) {
                memcpy((float*)dst, (float*)src + j, sizeof(float));
            }
        }
        return rv;
    } else {
        return src;
    }
}

static NPY_INLINE void
nan_FLOAT_vec(void *dst_in, const LINEARIZE_VDATA_t* data)
{
    float *dst = (float *) dst_in;

    int j;
    ptrdiff_t cs = data->strides/sizeof(float);
    for (j = 0; j < data->len; ++j) {
        *dst = s_nan;
        dst += cs;
    }
}


#line 471

static NPY_INLINE void *
linearize_DOUBLE_matrix(void *dst_in,
                         const void *src_in,
                         const LINEARIZE_DATA_t* data)
 {
    double *src = (double *) src_in;
    double *dst = (double *) dst_in;

    if (dst) {
        int i, j;
        double* rv = dst;
        fortran_int columns = (fortran_int)data->columns;
        fortran_int column_strides =
                (fortran_int)(data->column_strides/sizeof(double));
        fortran_int one = 1;
        for (i = 0; i < data->rows; i++) {
            if (column_strides > 0) {
                FNAME(dcopy)(&columns,
                              (void*)src, &column_strides,
                              (void*)dst, &one);
            }
            else if (column_strides < 0) {
                FNAME(dcopy)(&columns,
                              (void*)((double*)src + (columns-1)*column_strides),
                              &column_strides,
                              (void*)dst, &one);
            }
            else {
            /*
             * Zero stride has undefined behavior in some BLAS
             * implementations (e.g. OSX Accelerate), so do it
             * manually
             */
            for (j = 0; j < columns; ++j) {
                memcpy((double*)dst + j, (double*)src, sizeof(double));
            }
            }
            src += data->row_strides/sizeof(double);
            dst += data->output_lead_dim;
        }
        return rv;
    } else {
        return src;
    }
}

static NPY_INLINE void *
delinearize_DOUBLE_matrix(void *dst_in,
                        const void *src_in,
                        const LINEARIZE_DATA_t* data)
{
    double *src = (double *) src_in;
    double *dst = (double *) dst_in;

    if (src) {
        int i;
        double *rv = src;
        fortran_int columns = (fortran_int)data->columns;
        fortran_int column_strides =
          (fortran_int)(data->column_strides/sizeof(double));
        fortran_int one = 1;
        for (i = 0; i < data->rows; i++) {
            if (column_strides > 0) {
                FNAME(dcopy)(&columns,
                              (void*)src, &one,
                              (void*)dst, &column_strides);
            }
            else if (column_strides < 0) {
                FNAME(dcopy)(&columns,
                              (void*)src, &one,
                              (void*)((double*)dst + (columns-1)*column_strides),
                              &column_strides);
            }
            else {
              /*
               * Zero stride has undefined behavior in some BLAS
               * implementations (e.g. OSX Accelerate), so do it
               * manually
               */
                if (columns > 0) {
                    memcpy((double*)dst,
                           (double*)src + (columns-1),
                           sizeof(double));
              }
            }
            src += data->output_lead_dim;
            dst += data->row_strides/sizeof(double);
        }
        return rv;
    } else {
        return src;
    }
}

static NPY_INLINE void *
delinearize_DOUBLE_triu(void *dst_in,
                        const void *src_in,
                        const LINEARIZE_DATA_t* data)
{
   double *src = (double *) src_in;
   double *dst = (double *) dst_in;

   if (src) {
       int i;
        double *rv = src;
        fortran_int columns = (fortran_int)data->columns;
        fortran_int column_strides =
            (fortran_int)(data->column_strides/sizeof(double));
        fortran_int one = 1;
        for (i = 0; i < data->rows; i++) {
            fortran_int n = fortran_int_min(i + one, columns);
            if (column_strides > 0) {
                FNAME(dcopy)(&n,
                              (void*)src, &one,
                              (void*)dst, &column_strides);
            }
            else if (column_strides < 0) {
                FNAME(dcopy)(&n,
                              (void*)src, &one,
                              (void*)((double*)dst + (n-1)*column_strides),
                              &column_strides);
            }
            else {
               /*
                * Zero stride has undefined behavior in some BLAS
                * implementations (e.g. OSX Accelerate), so do it
                * manually
                */
                if (columns > 0) {
                    memcpy((double*)dst,
                           (double*)src + (columns-1),
                           sizeof(double));
                }
            }
            src += data->output_lead_dim;
            dst += data->row_strides/sizeof(double);
        }

        return rv;
    } else {
        return src;
    }
}

static NPY_INLINE void
nan_DOUBLE_matrix(void *dst_in, const LINEARIZE_DATA_t* data)
{
    double *dst = (double *) dst_in;

    int i, j;
    ptrdiff_t cs = data->column_strides/sizeof(double);
    for (i = 0; i < data->rows; i++) {
        double *cp = dst;
        for (j = 0; j < data->columns; ++j) {
            *cp = d_nan;
            cp += cs;
        }
        dst += data->row_strides/sizeof(double);
    }
}

static NPY_INLINE void
zero_DOUBLE_matrix(void *dst_in, const LINEARIZE_DATA_t* data)
{
    double *dst = (double *) dst_in;

    int i, j;
    ptrdiff_t cs = data->column_strides/sizeof(double);
    for (i = 0; i < data->rows; i++) {
        double *cp = dst;
        for (j = 0; j < data->columns; ++j) {
            *cp = d_zero;
            cp += cs;
        }
        dst += data->row_strides/sizeof(double);
    }
}

static NPY_INLINE void *
delinearize_DOUBLE_vec(void *dst_in,
                     void *src_in,
                     const LINEARIZE_VDATA_t *data)
{
    double *src = (double *) src_in;
    double *dst = (double *) dst_in;

    if (dst) {
        double* rv = dst;
        fortran_int len = (fortran_int)data->len;
        fortran_int strides = (fortran_int)(data->strides/sizeof(double));
        fortran_int one = 1;
        if (strides > 0) {
            FNAME(dcopy)(&len,
                          (void*)src, &one,
                          (void*)dst, &strides);
        }
        else if (strides < 0) {
            FNAME(dcopy)(&len,
                          (void*)((double*)src + (len-1)*strides),
                          &one,
                          (void*)dst, &strides);
        }
        else {
            /*
             * Zero stride has undefined behavior in some BLAS
             * implementations (e.g. OSX Accelerate), so do it
             * manually
             */
            int j;
            for (j = 0; j < len; ++j) {
                memcpy((double*)dst, (double*)src + j, sizeof(double));
            }
        }
        return rv;
    } else {
        return src;
    }
}

static NPY_INLINE void
nan_DOUBLE_vec(void *dst_in, const LINEARIZE_VDATA_t* data)
{
    double *dst = (double *) dst_in;

    int j;
    ptrdiff_t cs = data->strides/sizeof(double);
    for (j = 0; j < data->len; ++j) {
        *dst = d_nan;
        dst += cs;
    }
}



/*
 *****************************************************************************
 **                         QR DECOMPOSITION                                **
 *****************************************************************************
 */

char *qr_m_signature = "(m,n)->(m,m),(m,n)";  // m<n
char *qr_n_signature = "(m,n)->(m,n),(n,n)";  // m>n

typedef struct geqrf_params_struct
{
    void *A; /* A is (M,N) of base type */
    void *T; /* X is (K,) of base type */
    void *WR; /* WR is (N*B,) of base type, work for _geqrf */
    void *WQ; /* WQ is (N*B,) of base type, work for _orgqr */

    fortran_int M;
    fortran_int N;
    fortran_int K;
    fortran_int NC;
    fortran_int LDA;
    fortran_int LWR; /* LWR is lwork for _geqrf */
    fortran_int LWQ; /* LWQ is lwork for _orgqr */
    fortran_int INFO;
} GEQRF_PARAMS_t;

/* ************************************************
* Calling BLAS/Lapack functions _geqrf and _orgqr
*************************************************** */

static NPY_INLINE void
call_dgeqrf(GEQRF_PARAMS_t *params)
{
    // A,T are modified by ?GEQRF to carry QR info
    LAPACK(dgeqrf)(&params->M, &params->N, params->A, &params->LDA,
                    params->T, params->WR, &params->LWR, &params->INFO);
}

static NPY_INLINE void
call_dorgqr(GEQRF_PARAMS_t *params)
{
    // A is modified by ?ORGQR to carry Q
    LAPACK(dorgqr)(&params->M, &params->NC, &params->K, params->A, &params->LDA,
                    params->T, params->WQ, &params->LWQ, &params->INFO);
}

/* *************************************************************************
 * Initialize the parameters to use in the lapack functions _geqrf &  _orgqr
 * Handles buffer allocation
 ************************************************************************** */
static NPY_INLINE int
init_DOUBLE_qr(GEQRF_PARAMS_t *params, fortran_int M, fortran_int N, fortran_int NC)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *b, *c, *d;
    size_t safe_M = M;
    size_t safe_N = N;
    size_t safe_NC = fortran_int_max(NC, N);
    fortran_int ld = fortran_int_max(M, 1);
    fortran_int K = fortran_int_min(M, N);
    size_t safe_K = K;
    fortran_doublereal work_size;
    mem_buff = malloc(safe_M * safe_NC * sizeof(fortran_doublereal)
                   + safe_K * sizeof(fortran_doublereal));
    if (!mem_buff) {
        goto error;
    }
    a = mem_buff;
    b = a + safe_M * safe_NC * sizeof(fortran_doublereal);

    params->A = a;
    params->T = b;
    params->WR = &work_size;
    params->WQ = &work_size;
    params->M = M;
    params->N = N;
    params->K = K;
    params->NC = NC;
    params->LDA = ld;
    params->LWR = -1;
    params->LWQ = -1;
    params->INFO = 0;

    call_dgeqrf(params);
    if (params->INFO < 0) {
        goto error;
    }
    fortran_int LWR = (fortran_int)work_size;
    size_t safe_LWR = LWR;

    call_dorgqr(params);
    if (params->INFO < 0) {
        goto error;
    }
    fortran_int LWQ = (fortran_int)work_size;
    size_t safe_LWQ = LWQ;

    mem_buff2 = malloc(safe_LWR * sizeof(fortran_doublereal)
                    + safe_LWQ * sizeof(fortran_doublereal));
    if (!mem_buff2) {
        goto error;
    }
    c = mem_buff2;
    d = c + safe_LWR * sizeof(fortran_doublereal);

    params->WR = c;
    params->WQ = d;
    params->LWR = LWR;
    params->LWQ = LWQ;
    params->INFO = 0;

    return 1;
  error:
    free(mem_buff);
    free(mem_buff2);
    memset(params, 0, sizeof(*params));
    PyErr_NoMemory();

    return 0;
}

  /* ********************
  * Deallocate buffer
  *********************** */

static NPY_INLINE void
release_DOUBLE_qr(GEQRF_PARAMS_t *params)
{
    /* 1st memory block base is in A, second in WR */
    free(params->A);
    free(params->WR);
    memset(params, 0, sizeof(*params));
}


 /* ********************
 * Inner GUfunc loop
 *********************** */

static int
do_DOUBLE_qr(const void *A, void *Q, void *R,
             GEQRF_PARAMS_t *params,
             const LINEARIZE_DATA_t *a_in,  LINEARIZE_DATA_t *q_out,
             const LINEARIZE_DATA_t *r_out)
{
    // copy input to buffer
    linearize_DOUBLE_matrix(params->A, A, a_in);
    // QR decompose
    call_dgeqrf(params);
    if (params->INFO < 0) {
      return 1;
    }
    // Zero out R
    zero_DOUBLE_matrix(R, r_out);
    // Copy R from buffer & triangularise
    delinearize_DOUBLE_triu(R, params->A, r_out);
    // Build Q
    call_dorgqr(params);
    if (params->INFO < 0) {
      return 1;
    }
    // Copy Q from buffer
    delinearize_DOUBLE_matrix(Q, params->A, q_out);
    return 0;
}

static void
DOUBLE_qr(char **args, npy_intp *dimensions, npy_intp *steps, int complete)
{
INIT_OUTER_LOOP_3
    npy_intp len_m = *dimensions++;  // rows
    npy_intp len_n = *dimensions++;  // columns
    npy_intp stride_a_m = *steps++;  // rows
    npy_intp stride_a_n = *steps++;
    npy_intp stride_q_m = *steps++;  // rows
    npy_intp stride_q_k = *steps++;
    npy_intp stride_r_k = *steps++;  // rows
    npy_intp stride_r_n = *steps++;
    int error_occurred = get_fp_invalid_and_clear();
    GEQRF_PARAMS_t params;
    LINEARIZE_DATA_t a_in, q_out, r_out;
    npy_intp len_nc = complete ? len_m : len_n;

    if(len_m < len_nc) {//signature demands a wide matrix for q, which is impossible for qr_n.
        // PyErr_SetString(PyExc_ValueError, "qr_n can only be called when m >= n.");
        error_occurred = 1;
        init_linearize_data(&q_out, len_nc, len_m, stride_q_k, stride_q_m);
        init_linearize_data_ex(&r_out, len_n, len_nc, stride_r_n, stride_r_k, len_m);
        nan_DOUBLE_matrix(args[1], &q_out);
        nan_DOUBLE_matrix(args[2], &r_out);
    } else {
        if(init_DOUBLE_qr(&params, len_m, len_n, len_nc)){
            init_linearize_data(&a_in, len_n, len_m, stride_a_n, stride_a_m);
            init_linearize_data(&q_out, len_nc, len_m, stride_q_k, stride_q_m);
            init_linearize_data_ex(&r_out, len_n, len_nc, stride_r_n, stride_r_k, len_m);

            BEGIN_OUTER_LOOP_3
                int not_ok;
                not_ok = do_DOUBLE_qr(args[0], args[1], args[2], &params,
                                   &a_in, &q_out, &r_out);
                if (not_ok) {
                    error_occurred = 1;
                    nan_DOUBLE_matrix(args[1], &q_out);
                    nan_DOUBLE_matrix(args[2], &r_out);
                }
            END_OUTER_LOOP
            release_DOUBLE_qr(&params);
        }
    }
    set_fp_invalid_or_clear(error_occurred);
}

static void
DOUBLE_qr_m(char **args, npy_intp *dimensions, npy_intp *steps,
        void *NPY_UNUSED(func))
{
    DOUBLE_qr(args, dimensions, steps, 1);
}

static void
DOUBLE_qr_n(char **args, npy_intp *dimensions, npy_intp *steps,
        void *NPY_UNUSED(func))
{
    DOUBLE_qr(args, dimensions, steps, 0);
}

/*
******************************************************************************
**                                SOLVE                                     **
******************************************************************************
*/

char *solve_signature = "(n,n),(n,nrhs)->(n,nrhs)";

typedef struct gesv_params_struct
{
    void *A; /* A is (N,N) of base type */
    void *B; /* B is (N,NRHS) of base type */
    fortran_int *IPIV; /* IPIV is (N,) of int type, work for _geqrf */

    fortran_int N;
    fortran_int NRHS;
    fortran_int LDA;
    fortran_int LDB;
    fortran_int INFO;
} GESV_PARAMS_t;

/* ************************************************
 * Calling BLAS/Lapack functions _gesv
 *************************************************** */

static NPY_INLINE void
call_dgesv(GESV_PARAMS_t *params)
{
    // A,B are modified by ?GESV to carry LU info & X
    LAPACK(dgesv)(&params->N, &params->NRHS, params->A, &params->LDA,
                   params->IPIV, params->B, &params->LDB, &params->INFO);
}

/* *************************************************************************
 * Initialize the parameters to use in the lapack functions _gesv
 * Handles buffer allocation
 ************************************************************************** */
static NPY_INLINE int
init_dgesv(GESV_PARAMS_t *params, fortran_int N, fortran_int NRHS)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *a, *b, *c;
    size_t safe_N = N;
    size_t safe_NRHS = NRHS;
    fortran_int lda = fortran_int_max(N, 1);
    fortran_int ldb = fortran_int_max(N, 1);
    mem_buff = malloc(safe_N * safe_N * sizeof(fortran_doublereal)
                    + safe_N * safe_NRHS * sizeof(fortran_doublereal)
                    + safe_N * sizeof(fortran_int));
    if (!mem_buff) {
        goto error;
    }
    a = mem_buff;
    b = a + safe_N * safe_N * sizeof(fortran_doublereal);
    c = b + safe_N * safe_NRHS * sizeof(fortran_doublereal);

    params->A = a;
    params->B = b;
    params->IPIV = (fortran_int*)c;
    params->N = N;
    params->NRHS = NRHS;
    params->LDA = lda;
    params->LDB = ldb;
    params->INFO = 0;

    return 1;

  error:
    free(mem_buff);
    memset(params, 0, sizeof(*params));
    PyErr_NoMemory();

    return 0;
}

/* ********************
* Deallocate buffer
*********************** */

static NPY_INLINE void
release_dgesv(GESV_PARAMS_t *params)
{
    /* 1st memory block base is in A */
    free(params->A);
    memset(params, 0, sizeof(*params));
}


/* ********************
* Inner GUfunc loop
*********************** */

static void
DOUBLE_solve(char **args, npy_intp *dimensions, npy_intp *steps,
 void *NPY_UNUSED(func))
{
INIT_OUTER_LOOP_3
    npy_intp len_n = *dimensions++;  // rows
    npy_intp len_nrhs = *dimensions++;  // columns
    npy_intp stride_a_r = *steps++;  // rows
    npy_intp stride_a_c = *steps++;
    npy_intp stride_b_r = *steps++;  // rows
    npy_intp stride_b_c = *steps++;
    npy_intp stride_x_r = *steps++;  // rows
    npy_intp stride_x_c = *steps++;
    int error_occurred = get_fp_invalid_and_clear();
    GESV_PARAMS_t params;
    LINEARIZE_DATA_t a_in, b_in, x_out;

    init_linearize_data(&a_in, len_n, len_n, stride_a_c, stride_a_r);
    init_linearize_data(&b_in, len_nrhs, len_n, stride_b_c, stride_b_r);
    init_linearize_data(&x_out, len_nrhs, len_n, stride_x_c, stride_x_r);

    if(init_dgesv(&params, len_n, len_nrhs)){
        BEGIN_OUTER_LOOP_3
            int not_ok;
            linearize_DOUBLE_matrix(params.A, args[0], &a_in);
            linearize_DOUBLE_matrix(params.B, args[1], &b_in);
            call_dgesv(&params);
            not_ok = params.INFO;
            if (not_ok) {
                error_occurred = 1;
                nan_DOUBLE_matrix(args[2], &x_out);
            } else {
                delinearize_DOUBLE_matrix(args[2], params.B, &x_out);
            }
        END_OUTER_LOOP
        release_dgesv(&params);
    }
    set_fp_invalid_or_clear(error_occurred);
}

 /*
 ******************************************************************************
 **                                LSTSQ                                     **
 ******************************************************************************
 */

char *lstsq_signature = "(m,n),(m,nrhs)->(n,nrhs)";

typedef struct gels_params_struct
{
    void *A; /* A is (N,N) of base type */
    void *B; /* B is (N,NRHS) of base type */
    void *W; /* W is (LW,) of base type, work for _geqrf */
    void *S; /* S is (MN,) of base type, work for _geqrf */
    void *RCOND; /* RCOND is scalar of base type */
    fortran_int *RANK; /* RANK is scalar of int type */
    fortran_int *IW; /* IW is (LIW,) of int type */

    fortran_int M;
    fortran_int N;
    fortran_int NRHS;
    fortran_int LDA;
    fortran_int LDB;
    fortran_int LW;
    fortran_int INFO;
} GELS_PARAMS_t;

 /* ************************************************
  * Calling BLAS/Lapack functions _gelsd
  *************************************************** */

static NPY_INLINE void
call_dgelsd(GELS_PARAMS_t *params)
{
    // A,B are modified by ?GELS to carry LU info & X
    LAPACK(dgelsd)(&params->M, &params->N, &params->NRHS,
                   params->A, &params->LDA, params->B, &params->LDB,
                   params->S, params->RCOND, params->RANK,
                   params->W, &params->LW, params->IW, &params->INFO);
}

 /* *************************************************************************
  * Initialize the parameters to use in the lapack functions _gelsd
  * Handles buffer allocation
  ************************************************************************** */
static NPY_INLINE int
init_dgelsd(GELS_PARAMS_t *params, fortran_int M, fortran_int N, fortran_int NRHS)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *b, *c, *d, *e;
    size_t safe_M = M;
    size_t safe_N = N;
    size_t safe_NRHS = NRHS;
    fortran_int MNx = fortran_int_max(M, N);
    size_t safe_MNx = MNx;
    fortran_int MNn = fortran_int_min(M, N);
    size_t safe_MNn = MNn;
    fortran_int lda = fortran_int_max(M, 1);
    fortran_int ldb = fortran_int_max(MNx, 1);
    fortran_doublereal work_size;
    fortran_int iwork_size;
    mem_buff = malloc(safe_M * safe_N * sizeof(fortran_doublereal)
                    + safe_MNx * safe_NRHS * sizeof(fortran_doublereal)
                    + safe_MNn  * sizeof(fortran_doublereal)
                    + sizeof(fortran_doublereal) + sizeof(fortran_int));
    if (!mem_buff) {
        goto error;
    }
    a = mem_buff;
    b = a + safe_M * safe_N * sizeof(fortran_doublereal);
    c = b + safe_MNx * safe_NRHS * sizeof(fortran_doublereal);
    d = c + safe_MNn * sizeof(fortran_doublereal);
    e = d + sizeof(fortran_doublereal);

    params->A = a;
    params->B = b;
    params->S = c;
    params->RCOND = d;
    params->RANK = (fortran_int*)e;
    params->W = &work_size;
    params->IW = &iwork_size;
    params->M = M;
    params->N = N;
    params->NRHS = NRHS;
    params->LDA = lda;
    params->LDB = ldb;
    params->LW = -1;
    params->INFO = 0;

    *(fortran_doublereal *)params->RCOND = MNx * d_eps;

    call_dgelsd(params);
    if (params->INFO) {
        goto error;
    }
    fortran_int LW = (fortran_int)work_size;
    size_t safe_LW = LW;
    fortran_int LIW = iwork_size;
    size_t safe_LIW = LIW;

    mem_buff2 = malloc(safe_LW * sizeof(fortran_doublereal)
                    + safe_LIW * sizeof(fortran_int));
    if (!mem_buff2) {
        goto error;
    }
    a = mem_buff2;
    b = a + safe_LW * sizeof(fortran_doublereal);

    params->W = a;
    params->IW = (fortran_int*)b;
    params->LW = LW;

    return 1;

  error:
    free(mem_buff);
    free(mem_buff2);
    memset(params, 0, sizeof(*params));
    PyErr_NoMemory();

    return 0;
}

/* ********************
* Deallocate buffer
*********************** */

static NPY_INLINE void
release_dgelsd(GELS_PARAMS_t *params)
{
   /* 1st memory block base is in A, second in W */
   free(params->A);
   free(params->W);
   memset(params, 0, sizeof(*params));
}

/* ********************
* Inner GUfunc loop
*********************** */

static void
DOUBLE_lstsq(char **args, npy_intp *dimensions, npy_intp *steps,
void *NPY_UNUSED(func))
{
INIT_OUTER_LOOP_3
    npy_intp len_m = *dimensions++;  // rows of a, b
    npy_intp len_n = *dimensions++;  // columns of a, rows of x
    npy_intp len_nrhs = *dimensions++;  // columns of x, b
    npy_intp stride_a_r = *steps++;  // rows
    npy_intp stride_a_c = *steps++;
    npy_intp stride_b_r = *steps++;  // rows
    npy_intp stride_b_c = *steps++;
    npy_intp stride_x_r = *steps++;  // rows
    npy_intp stride_x_c = *steps++;
    int error_occurred = get_fp_invalid_and_clear();
    GELS_PARAMS_t params;
    LINEARIZE_DATA_t a_in, b_in, x_out;
    npy_intp len_mn = len_m > len_n ? len_m : len_n;

    init_linearize_data(&a_in, len_n, len_m, stride_a_c, stride_a_r);
    init_linearize_data_ex(&b_in, len_nrhs, len_m, stride_b_c, stride_b_r, len_mn);
    init_linearize_data_ex(&x_out, len_nrhs, len_n, stride_x_c, stride_x_r, len_mn);

    if(init_dgelsd(&params, len_m, len_n, len_nrhs)){
        BEGIN_OUTER_LOOP_3
            int not_ok;
            linearize_DOUBLE_matrix(params.A, args[0], &a_in);
            linearize_DOUBLE_matrix(params.B, args[1], &b_in);
            call_dgelsd(&params);
            not_ok = params.INFO;
            if (not_ok) {
                error_occurred = 1;
                nan_DOUBLE_matrix(args[2], &x_out);
            } else {
                delinearize_DOUBLE_matrix(args[2], params.B, &x_out);
            }
        END_OUTER_LOOP
        release_dgelsd(&params);
    }
    set_fp_invalid_or_clear(error_occurred);
}

/*
******************************************************************************
**                              EIGVALSH                                    **
******************************************************************************
*/

char *eigvalsh_signature = "(n,n)->(n)";

typedef struct syevd_params_struct
{
    void *A; /* A is (N,N) of base type */
    void *E; /* B is (N,) of base type */
    void *W; /* W is (LW,) of base type, work for _geqrf */
    fortran_int *IW; /* IW is (LIW,) of int type */

    fortran_int N;
    fortran_int LDA;
    fortran_int LW;
    fortran_int LIW;
    fortran_int INFO;
    char JOBZ;
    char UPLO;
} SYEVD_PARAMS_t;

/* ************************************************
 * Calling BLAS/Lapack functions _syevd
 *************************************************** */

static NPY_INLINE void
call_dsyevd(SYEVD_PARAMS_t *params)
{
    // A,B are modified by ?GELS to carry LU info & X
    LAPACK(dsyevd)(&params->JOBZ, &params->UPLO, &params->N,
                params->A, &params->LDA, params->E,
                params->W, &params->LW, params->IW, &params->LIW, &params->INFO);
}

/* *************************************************************************
 * Initialize the parameters to use in the lapack functions _syevd
 * Handles buffer allocation
 ************************************************************************** */
static NPY_INLINE int
init_dsyevd(SYEVD_PARAMS_t *params, fortran_int N)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *b, *c, *d;
    size_t safe_N = N;
    fortran_int lda = fortran_int_max(N, 1);
    fortran_doublereal work_size;
    fortran_int iwork_size;
    mem_buff = malloc(safe_N * safe_N * sizeof(fortran_doublereal)
                    + safe_N * sizeof(fortran_doublereal));
    if (!mem_buff) {
        goto error;
    }
    a = mem_buff;
    b = a + safe_N * safe_N * sizeof(fortran_doublereal);

    params->A = a;
    params->E = b;
    params->W = &work_size;
    params->IW = &iwork_size;
    params->N = N;
    params->LDA = lda;
    params->LW = -1;
    params->LIW = -1;
    params->INFO = 0;
    params->JOBZ = 'N';
    params->UPLO = 'U';

    call_dsyevd(params);
    if (params->INFO) {
        goto error;
    }
    fortran_int LW = (fortran_int)work_size;
    size_t safe_LW = LW;
    fortran_int LIW = iwork_size;
    size_t safe_LIW = LIW;

    mem_buff2 = malloc(safe_LW * sizeof(fortran_doublereal)
                    + safe_LIW * sizeof(fortran_int));
    if (!mem_buff2) {
        goto error;
    }
    c = mem_buff2;
    d = a + safe_LW * sizeof(fortran_doublereal);

    params->W = c;
    params->IW = (fortran_int*)d;
    params->LW = LW;
    params->LIW = LIW;

    return 1;

  error:
    free(mem_buff);
    free(mem_buff2);
    memset(params, 0, sizeof(*params));
    PyErr_NoMemory();

    return 0;
}

/* ********************
* Deallocate buffer
*********************** */

static NPY_INLINE void
release_dsyevd(SYEVD_PARAMS_t *params)
{
    /* 1st memory block base is in A, second in W */
    free(params->A);
    free(params->W);
    memset(params, 0, sizeof(*params));
}

/* ********************
* Inner GUfunc loop
*********************** */

static void
DOUBLE_eigvalsh(char **args, npy_intp *dimensions, npy_intp *steps,
void *NPY_UNUSED(func))
{
INIT_OUTER_LOOP_2
    npy_intp len_n = *dimensions++;  // columns of a, rows of a
    npy_intp stride_a_r = *steps++;  // rows
    npy_intp stride_a_c = *steps++;
    npy_intp stride_e = *steps++;  //
    int error_occurred = get_fp_invalid_and_clear();
    SYEVD_PARAMS_t params;
    LINEARIZE_DATA_t a_in;
    LINEARIZE_VDATA_t e_out;

    init_linearize_data(&a_in, len_n, len_n, stride_a_c, stride_a_r);
    init_linearize_vdata(&e_out, len_n, stride_e);

    if(init_dsyevd(&params, len_n)){
        BEGIN_OUTER_LOOP_2
            int not_ok;
            linearize_DOUBLE_matrix(params.A, args[0], &a_in);
            call_dsyevd(&params);
            not_ok = params.INFO;
            if (not_ok) {
                error_occurred = 1;
                nan_DOUBLE_vec(args[1], &e_out);
            } else {
                delinearize_DOUBLE_vec(args[1], params.E, &e_out);
            }
        END_OUTER_LOOP
        release_dsyevd(&params);
    }
    set_fp_invalid_or_clear(error_occurred);
}

/*
******************************************************************************
**                              SINGVALS                                    **
******************************************************************************
*/

char *singvals_m_signature = "(m,n)->(m)";
char *singvals_n_signature = "(m,n)->(n)";

typedef struct gesdd_params_struct
{
    void *A; /* A is (N,N) of base type */
    void *S; /* B is (N,) of base type */
    void *U; /* B is (N,) of base type */
    void *V; /* B is (N,) of base type */
    void *W; /* W is (LW,) of base type, work for _geqrf */
    fortran_int *IW; /* IW is (LIW,) of int type */

    fortran_int M;
    fortran_int N;
    fortran_int LDA;
    fortran_int LDU;
    fortran_int LDV;
    fortran_int LW;
    fortran_int LIW;
    fortran_int INFO;
    char JOBZ;
} GESDD_PARAMS_t;


/* ************************************************
 * Calling BLAS/Lapack functions _gesdd
 *************************************************** */

static NPY_INLINE void
call_dgesdd(GESDD_PARAMS_t *params)
{
    // S,A are modified by ?GESDD to carry singvals & ?
    LAPACK(dgesdd)(&params->JOBZ, &params->M, &params->N,
                params->A, &params->LDA, params->S,
                params->U, &params->LDU, params->V, &params->LDV,
                params->W, &params->LW, params->IW, &params->INFO);
}

/* *************************************************************************
 * Initialize the parameters to use in the lapack functions _gesdd
 * Handles buffer allocation
 ************************************************************************** */
static NPY_INLINE int
init_dgesdd(GESDD_PARAMS_t *params, fortran_int M, fortran_int N)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *b, *c, *d;
    size_t safe_M = M;
    size_t safe_N = N;
    fortran_int MN = fortran_int_min(M, N);
    size_t safe_MN = MN;
    fortran_int lda = fortran_int_max(M, 1);
    fortran_doublereal work_size;
    fortran_int iwork_size;
    mem_buff = malloc(safe_M * safe_N * sizeof(fortran_doublereal)
                    + safe_MN * sizeof(fortran_doublereal));
    if (!mem_buff) {
        goto error;
    }
    a = mem_buff;
    b = a + safe_M * safe_N * sizeof(fortran_doublereal);

    params->A = a;
    params->S = b;
    params->U = NULL; // unused
    params->V = NULL; // unused
    params->W = &work_size;
    params->IW = &iwork_size;
    params->M = M;
    params->N = N;
    params->LDA = lda;
    params->LDU = lda; // unused
    params->LDV = lda; // unused
    params->LW = -1;
    params->INFO = 0;
    params->JOBZ = 'N';

    call_dgesdd(params);
    if (params->INFO) {
        goto error;
    }
    fortran_int LW = (fortran_int)work_size;
    size_t safe_LW = LW;
    fortran_int LIW = iwork_size;
    size_t safe_LIW = LIW;

    mem_buff2 = malloc(safe_LW * sizeof(fortran_doublereal)
                    + safe_LIW * sizeof(fortran_int));
    if (!mem_buff2) {
        goto error;
    }
    c = mem_buff2;
    d = a + safe_LW * sizeof(fortran_doublereal);

    params->W = c;
    params->IW = (fortran_int*)d;
    params->LW = LW;

    return 1;

  error:
    free(mem_buff);
    free(mem_buff2);
    memset(params, 0, sizeof(*params));
    PyErr_NoMemory();

    return 0;
}

/* ********************
* Deallocate buffer
*********************** */

static NPY_INLINE void
release_dgesdd(GESDD_PARAMS_t *params)
{
    /* 1st memory block base is in A, second in W */
    free(params->A);
    free(params->W);
    memset(params, 0, sizeof(*params));
}


/* ********************
* Inner GUfunc loop
*********************** */

static void
DOUBLE_singvals(char **args, npy_intp *dimensions, npy_intp *steps,
void *NPY_UNUSED(func))
{
INIT_OUTER_LOOP_2
    npy_intp len_m = *dimensions++;  // columns of a, rows of a
    npy_intp len_n = *dimensions++;  // columns of a, rows of a
    npy_intp stride_a_r = *steps++;  // rows
    npy_intp stride_a_c = *steps++;
    npy_intp stride_s = *steps++;  //
    int error_occurred = get_fp_invalid_and_clear();
    GESDD_PARAMS_t params;
    LINEARIZE_DATA_t a_in;
    LINEARIZE_VDATA_t s_out;
    npy_intp len_k = len_m < len_n ? len_m : len_n;

    init_linearize_data(&a_in, len_n, len_m, stride_a_c, stride_a_r);
    init_linearize_vdata(&s_out, len_k, stride_s);

    if(init_dgesdd(&params, len_m, len_n)){
        BEGIN_OUTER_LOOP_2
            int not_ok;
            linearize_DOUBLE_matrix(params.A, args[0], &a_in);
            call_dgesdd(&params);
            not_ok = params.INFO;
            if (not_ok) {
                error_occurred = 1;
                nan_DOUBLE_vec(args[1], &s_out);
            } else {
                delinearize_DOUBLE_vec(args[1], params.S, &s_out);
            }
        END_OUTER_LOOP
        release_dgesdd(&params);
    }
    set_fp_invalid_or_clear(error_occurred);
}



/*
*****************************************************************************
**                             UFUNC DEFINITION                            **
*****************************************************************************
*/

static void *null_data_1[] = { (void *)NULL, (void *)NULL };
static char ufn_types_1_3[] = { NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
                                NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE };
static char ufn_types_1_2[] = { NPY_FLOAT, NPY_FLOAT,
                                NPY_DOUBLE, NPY_DOUBLE };


static PyUFuncGenericFunction qr_m_functions[] =
                         { FLOAT_qr_m, DOUBLE_qr_m};

static PyUFuncGenericFunction qr_n_functions[] =
                         { FLOAT_qr_n, DOUBLE_qr_n};

static PyUFuncGenericFunction solve_functions[] =
                         { FLOAT_solve, DOUBLE_solve};

static PyUFuncGenericFunction lstsq_functions[] =
                         { FLOAT_lstsq, DOUBLE_lstsq};

static PyUFuncGenericFunction eigvalsh_functions[] =
                         { FLOAT_eigvalsh, DOUBLE_eigvalsh};

static PyUFuncGenericFunction singvals_functions[] =
                         { FLOAT_singvals, DOUBLE_singvals};



static int
addUfuncs(PyObject *dictionary) {
    PyObject *f;

    f = PyUFunc_FromFuncAndDataAndSignature(qr_m_functions,
            null_data_1, ufn_types_1_3, 1, 1, 2, PyUFunc_None,
            "qr_m", qr__doc__, 0, qr_m_signature);
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "qr_m", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(qr_n_functions,
            null_data_1, ufn_types_1_3, 1, 1, 2, PyUFunc_None,
            "qr_n", qr__doc__, 0, qr_n_signature);
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "qr_n", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(solve_functions,
            null_data_1, ufn_types_1_3, 1, 2, 1, PyUFunc_None,
            "solve", solve__doc__, 0, solve_signature);
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "solve", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(lstsq_functions,
            null_data_1, ufn_types_1_3, 1, 2, 1, PyUFunc_None,
            "lstsq", lstsq__doc__, 0, lstsq_signature);
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "lstsq", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(eigvalsh_functions,
            null_data_1, ufn_types_1_2, 1, 1, 1, PyUFunc_None,
            "eigvalsh", eigvalsh__doc__, 0, eigvalsh_signature);
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "eigvalsh", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(singvals_functions,
            null_data_1, ufn_types_1_2, 1, 1, 1, PyUFunc_None,
            "singvals_m", singvals__doc__, 0, singvals_m_signature);
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "singvals_m", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndDataAndSignature(singvals_functions,
            null_data_1, ufn_types_1_2, 1, 1, 1, PyUFunc_None,
            "singvals_n", singvals__doc__, 0, singvals_n_signature);
    if (f == NULL) {
        return -1;
    }
    PyDict_SetItemString(dictionary, "singvals_n", f);
    Py_DECREF(f);

    return 0;
}

/*
*****************************************************************************
**               Module initialization stuff                               **
*****************************************************************************
*/

static PyMethodDef GUfuncs_LAPACK_Methods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_gufuncs_lapack",
    NULL,
    -1,
    GUfuncs_LAPACK_Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit__gufuncs_lapack(void)
{
    PyObject *m;
    PyObject *d;
    PyObject *version;

    init_constants();
    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    import_array();
    import_ufunc();

    d = PyModule_GetDict(m);

    version = PyString_FromString(gufuncs_lapack_version_string);
    PyDict_SetItemString(d, "__version__", version);
    Py_DECREF(version);

    /* Load the ufunc operators into the module's namespace */
    addUfuncs(d);

    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load _gufuncs_lapack module.");
        return NULL;
    }

    return m;
}
