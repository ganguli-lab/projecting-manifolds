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
52.  Includes
72.  Docstrings
137. BLAS/Lapack calling functions
165. Data rearrangement functions
309. PDIST_RATIO and CDIST_RATIO
555. NORM
638. MATMUL
780. Ufunc definition
802. Module initialization stuff
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

#include "gufunc_common.h"
#include "gufunc_fortran.h"

static const char* gufuncs_blas_version_string = "0.1.1";

/*
*****************************************************************************
**                   Doc string for Python functions                       **
*****************************************************************************
*/

PyDoc_STRVAR(pdist_ratio__doc__,
//"pdist_ratio(X: ndarray, P: ndarray) -> (drmax: float, drmin: float)\n\n"
"Maximum and minimum ratio of pair-wise distances squared beween correspoinding "
"pairs of points in two sets.\n\n"
"Parameters\n-----------\n"
"X: ndarray (P,N)\n"
"    Set of points between which we compute pairwise distances for the denominator. "
"    Each point is a row.\n"
"P: ndarray (P,M)\n"
"    Set of points between which we compute pairwise distances for the numerator.\n\n"
"Returns\n-------\n"
"drmax: float\n"
"    Maximum ratio of distances squared.\n"
"drmin: float\n"
"    Minimum ratio of distances squared.\n");

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
"    Set of points *to* which we compute pairwise distances for the numerator.\n\n"
"Returns\n-------\n"
"drmax: float\n"
"    Maximum ratio of distances squared.\n"
"drmin: float\n"
"    Minimum ratio of distances squared.\n");

PyDoc_STRVAR(matmul__doc__,
//"matmul(X: ndarray, Y: ndarray) -> (Z: ndarray)\n\n"
"Matrix-matrix product.\n\n"
"Parameters\n-----------\n"
"X: ndarray (M,N)\n"
"    Matrix multiplying from left.\n"
"Y: ndarray (N,P)\n"
"    Matrix multiplying from right.\n\n"
"Returns\n-------\n"
"Z: ndarray (M,P)\n"
"    Result of matrix multiplication.");

PyDoc_STRVAR(norm__doc__,
//"matmul(X: ndarray, Y: ndarray) -> (Z: ndarray)\n\n"
"Euclidean norm of a vector.\n\n"
"Parameters\n-----------\n"
"X: ndarray (...,N)\n"
"    Vector, or array of vectors.\n\n"
"Returns\n-------\n"
"Z: float\n"
"    Euclidean norm of X.");

/*
*****************************************************************************
*                    BLAS/LAPACK calling macros                             *
*****************************************************************************
*/

/* copy vector x into y */
extern int
FNAME(dcopy)(int *n,
             double *sx, int *incx,
             double *sy, int *incy);

/* y -> y + a x */
extern int
FNAME(daxpy)(int *n, double *da,
             double dx[], int *inc_x,
             double dy[], int *inc_y);

/* x -> sqrt(x'*x) */
extern double
FNAME(dnrm2)(int *n, double dx[], int *inc_x);

/* z -> a x*y + b z */
extern int
FNAME(dgemm)(char *transa, char *transb, int *m, int *n, int *k,
    double *alpha, double *a, int *lda, double *b, int *ldb,
    double *beta, double *c, int *ldc);

/*
*****************************************************************************
**                   Data rearrangement functions                          **
*****************************************************************************
*/

             /* rearranging of 2D matrices using blas */

static NPY_INLINE void *
linearize_DOUBLE_vec(void *dst_in,
                     const void *src_in,
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
                          (void*)src, &strides,
                          (void*)dst, &one);
        }
        else if (strides < 0) {
            FNAME(dcopy)(&len,
                          (void*)((double*)src + (len-1)*strides),
                          &strides,
                          (void*)dst, &one);
        }
        else {
            /*
             * Zero stride has undefined behavior in some BLAS
             * implementations (e.g. OSX Accelerate), so do it
             * manually
             */
            int j;
            for (j = 0; j < len; ++j) {
                memcpy((double*)dst + j, (double*)src, sizeof(double));
            }
        }
        return rv;
    } else {
        return src;
    }
}

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

/*
*******************************************************************************
**                      PDIST_RATIO and CDIST_RATIO                          **
*******************************************************************************
*/

typedef struct axpy_params_struct
{
    void *A; /* A is scalar of base type */
    void *X; /* X is (N,) of base type */
    void *Y; /* Y is (N,) of base type */

    fortran_int N;
    fortran_int INCX;
    fortran_int INCY;
} APXY_PARAMS_t;

/* *************************************************
* Calling BLAS/Lapack functions _apxy and _nrm2
*************************************************** */

static NPY_INLINE fortran_int
call_daxpy(APXY_PARAMS_t *params)
{
    LAPACK(daxpy)(&params->N, params->A,
    // Y is modified by ?APXY to carry difference
                    params->X, &params->INCX,
                    params->Y, &params->INCY);
}

static NPY_INLINE npy_double
call_dnrm2(APXY_PARAMS_t *params)
{
  // Y carries difference
    return LAPACK(dnrm2)(&params->N, params->Y, &params->INCY);
}

/* *****************************************************************
* Initialize the parameters to use in for the lapack function _axpy
* Handles buffer allocation
******************************************************************* */
static NPY_INLINE int
init_DOUBLE_dist(APXY_PARAMS_t *params, npy_intp N_in)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *a, *b;
    fortran_int N = (fortran_int)N_in;
    size_t safe_N = N_in;
    fortran_int ld = fortran_int_max(N, 1);
    mem_buff = malloc(safe_N * sizeof(fortran_doublereal)
                      + safe_N * sizeof(fortran_doublereal));
    if (!mem_buff) {
        goto error;
    }
    a = mem_buff;
    b = a + safe_N * sizeof(fortran_doublereal);

    params->A = &d_minus_one;
    params->X = a;
    params->Y = b;
    params->N = N;
    params->INCX = 1;
    params->INCY = 1;

    return 1;
 error:
    free(mem_buff);
    memset(params, 0, sizeof(*params));
    PyErr_NoMemory();

    return 0;
}

/********************
* Deallocate buffer
********************* */

static NPY_INLINE void
release_DOUBLE_dist(APXY_PARAMS_t *params)
{
    /* memory block base is in X */
    free(params->X);
    memset(params, 0, sizeof(*params));
}

/* *************************
* Inner GUfunc loop
**************************** */

static void
do_DOUBLE_dist(const void *Y, npy_double *dist, APXY_PARAMS_t *params,
            const LINEARIZE_VDATA_t *y_in)
{
    // linearize_DOUBLE_vec(params->X, X, x_in);
    linearize_DOUBLE_vec(params->Y, Y, y_in);
    call_daxpy(params);
    *dist = call_dnrm2(params);
}


static void
DOUBLE_pdist_ratio(char **args, npy_intp *dimensions, npy_intp *steps,
                   void *NPY_UNUSED(func))
{
INIT_OUTER_LOOP_4
    npy_intp len_d = *dimensions++;  // number of points
    npy_intp len_m = *dimensions++;  // dimensions of numerator
    npy_intp len_n = *dimensions++;  // dimensions of denominator
    npy_intp stride_num_d = *steps++;  // numerator
    npy_intp stride_m = *steps++;
    npy_intp stride_den_d = *steps++;  // denominator
    npy_intp stride_n = *steps++;
    npy_intp d1, d2;
    APXY_PARAMS_t nparams, dparams;
    LINEARIZE_VDATA_t num_in, den_in;

    init_linearize_vdata(&num_in, len_m, stride_m);
    init_linearize_vdata(&den_in, len_n, stride_n);

    if(init_DOUBLE_dist(&nparams, len_m)){
        if(init_DOUBLE_dist(&dparams, len_n)){

            BEGIN_OUTER_LOOP_4

                const char *ip_num_fr = args[0];  //  from-ptr: numerator
                const char *ip_den_fr = args[1];  //  from-ptr: denominator
                char *op1 = args[2], *op2 = args[3];
                npy_double dr_min = d_inf, dr_max = d_zero;  // running min/max distance ratio

                for (d1 = 0; d1 < len_d-1; d1++) {

                    linearize_DOUBLE_vec(nparams.X, ip_num_fr, &num_in);
                    linearize_DOUBLE_vec(dparams.X, ip_den_fr, &den_in);

                    const char *ip_num_to = ip_num_fr + stride_num_d;  //  to-ptr: numerator
                    const char *ip_den_to = ip_den_fr + stride_den_d;  //  to-ptr: denominator

                    for (d2 = d1 + 1; d2 < len_d; d2++) {

                        npy_double numerator, denominator;

                        do_DOUBLE_dist(ip_num_to, &numerator, &nparams, &num_in);
                        do_DOUBLE_dist(ip_den_to, &denominator, &dparams, &den_in);
                        // DOUBLE_dist(ip_num_fr, ip_num_to, &numerator, &nparams, &num_in, &num_in);
                        // DOUBLE_dist(ip_den_fr, ip_den_to, &denominator, &dparams, &den_in, &den_in);

                        npy_double ratio = numerator / denominator;
                        if (ratio < dr_min) dr_min = ratio;  // update running max/min
                        if (ratio > dr_max) dr_max = ratio;

                        ip_num_to += stride_num_d;  // next point: to
                        ip_den_to += stride_den_d;

                    }  // for d2
                    ip_num_fr += stride_num_d;  // next point: from
                    ip_den_fr += stride_den_d;

                }  // for d1
                *(npy_double *)op1 = dr_min;
                *(npy_double *)op2 = dr_max;

            END_OUTER_LOOP

            release_DOUBLE_dist(&dparams);
        }
        release_DOUBLE_dist(&nparams);
    }
}


static void
DOUBLE_cdist_ratio(char **args, npy_intp *dimensions, npy_intp *steps,
                   void *NPY_UNUSED(func))
{
INIT_OUTER_LOOP_6
    npy_intp len_fr_d = *dimensions++;  // number of points, from
    npy_intp len_m = *dimensions++;  // dimensions of numerator
    npy_intp len_to_d = *dimensions++;  // number of points, to
    npy_intp len_n = *dimensions++;  // dimensions of denominator
    npy_intp stride_num_fr_d = *steps++;  // numerator, from
    npy_intp stride_fr_m = *steps++;
    npy_intp stride_num_to_d = *steps++;  // numerator, to
    npy_intp stride_to_m = *steps++;
    npy_intp stride_den_fr_d = *steps++;  // denominator, from
    npy_intp stride_fr_n = *steps++;
    npy_intp stride_den_to_d = *steps++;  // denominator, to
    npy_intp stride_to_n = *steps++;
    npy_intp d1, d2;
    APXY_PARAMS_t nparams, dparams;
    LINEARIZE_VDATA_t num_fr_in, num_to_in, den_fr_in, den_to_in;

    init_linearize_vdata(&num_fr_in, len_m, stride_fr_m);
    init_linearize_vdata(&num_to_in, len_m, stride_to_m);
    init_linearize_vdata(&den_fr_in, len_n, stride_fr_n);
    init_linearize_vdata(&den_to_in, len_n, stride_to_n);

    if(init_DOUBLE_dist(&nparams, len_m)) {
        if(init_DOUBLE_dist(&dparams, len_n)) {

            BEGIN_OUTER_LOOP_6

                const char *ip_num_fr = args[0];  //  from-ptr: numerator
                const char *ip_den_fr = args[2];  //  from-ptr: denominator
                char *op1 = args[4], *op2 = args[5];
                npy_double dr_min = d_inf, dr_max = d_zero;  // min/max distance ratio

                for (d1 = 0; d1 < len_fr_d; d1++) {

                    linearize_DOUBLE_vec(nparams.X, ip_num_fr, &num_fr_in);
                    linearize_DOUBLE_vec(dparams.X, ip_den_fr, &den_fr_in);

                    const char *ip_num_to = args[1];  //  to-ptr: numerator
                    const char *ip_den_to = args[3];  //  to-ptr: denominator

                    for (d2 = d1 + 1; d2 < len_to_d; d2++) {

                        npy_double numerator, denominator;

                        do_DOUBLE_dist(ip_num_to, &numerator, &nparams, &num_to_in);
                        do_DOUBLE_dist(ip_den_to, &denominator, &dparams, &den_to_in);
                        // DOUBLE_dist(ip_num_fr, ip_num_to, &numerator, &nparams, &num_fr_in, &num_to_in);
                        // DOUBLE_dist(ip_den_fr, ip_den_to, &denominator, &dparams, &den_fr_in, &den_to_in);

                        npy_double ratio = numerator / denominator;
                        if (ratio < dr_min) dr_min = ratio;  // update running max/min
                        if (ratio > dr_max) dr_max = ratio;

                        ip_num_to += stride_num_to_d;  // next point: to
                        ip_den_to += stride_den_to_d;

                    }  // for d2
                    ip_num_fr += stride_num_fr_d;  // next point: from
                    ip_den_fr += stride_den_fr_d;

                }  // for d1
                *(npy_double *)op1 = dr_min;
                *(npy_double *)op2 = dr_max;

            END_OUTER_LOOP

            release_DOUBLE_dist(&dparams);
        }
        release_DOUBLE_dist(&nparams);
    }
}

/*
******************************************************************************
**                              NORM                                        **
******************************************************************************
*/

/* *****************************************************************
* Initialize the parameters to use in for the lapack function _axpy
* Handles buffer allocation
******************************************************************* */

static NPY_INLINE int
init_DOUBLE_nrm2(APXY_PARAMS_t *params, npy_intp N_in)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *a;
    fortran_int N = (fortran_int)N_in;
    size_t safe_N = N_in;
    fortran_int ld = fortran_int_max(N, 1);

    mem_buff = malloc(safe_N * sizeof(fortran_doublereal));
    if (!mem_buff) {
        goto error;
    }
    a = mem_buff;

    params->A = NULL;
    params->X = NULL;
    params->Y = a;
    params->N = N;
    params->INCX = 1;
    params->INCY = 1;

    return 1;
 error:
    free(mem_buff);
    memset(params, 0, sizeof(*params));
    PyErr_NoMemory();

    return 0;
}

/* *********************
* Deallocate buffer
************************ */

static NPY_INLINE void
release_DOUBLE_nrm2(APXY_PARAMS_t *params)
{
    /* memory block base is in Y */
    free(params->Y);
    memset(params, 0, sizeof(*params));
}

/* **************************
* Inner GUfunc loop
***************************** */

static void
DOUBLE_norm(char **args, npy_intp *dimensions, npy_intp *steps,
              void *NPY_UNUSED(func))
{
INIT_OUTER_LOOP_2

    npy_intp len_n = *dimensions++;  // dimensions of inner
    npy_intp stride_n = *steps++;  // 1st arg
    APXY_PARAMS_t params;
    LINEARIZE_VDATA_t y_in;

    init_linearize_vdata(&y_in, len_n, stride_n);

    if(init_DOUBLE_nrm2(&params, len_n)) {
        BEGIN_OUTER_LOOP_2

            linearize_DOUBLE_vec(params.Y, args[0], &y_in);
            *(npy_double *)args[1] = call_dnrm2(&params);

        END_OUTER_LOOP
        release_DOUBLE_nrm2(&params);
    }

}

/*
******************************************************************************
**                                  MATMUL                                  **
******************************************************************************
*/
typedef struct gemm_params_struct
{
  void *A; /* A is scalar of base type */
  void *B; /* B is scalar of base type */
  void *X; /* X is (M,K) of base type */
  void *Y; /* Y is (K,N) of base type */
  void *Z; /* Z is (M,N) of base type */

  fortran_int M;
  fortran_int N;
  fortran_int K;
  fortran_int LDX;
  fortran_int LDY;
  fortran_int LDZ;
  char TRANSX;
  char TRANSY;
} GEMM_PARAMS_t;

/* **************************************
* Calling BLAS/Lapack function _gemm
***************************************** */

static NPY_INLINE npy_double
call_dgemm(GEMM_PARAMS_t *params)
{
    LAPACK(dgemm)(&params->TRANSX, &params->TRANSY,
       &params->M, &params->N, &params->K,
       params->A,  params->X, &params->LDX, params->Y, &params->LDY,
       params->B, params->Z, &params->LDZ);
}

/* ******************************************************************
* Initialize the parameters to use in for the lapack function _gemm
* Handles buffer allocation
********************************************************************* */

static NPY_INLINE int
init_DOUBLE_matm(GEMM_PARAMS_t *params, npy_intp M_in, npy_intp N_in, npy_intp K_in)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *a, *b, *c;
    fortran_int M = (fortran_int)M_in;
    fortran_int N = (fortran_int)N_in;
    fortran_int K = (fortran_int)K_in;
    size_t safe_M = M_in;
    size_t safe_N = N_in;
    size_t safe_K = K_in;
    fortran_int ldx, ldy, ldz;

    // ldx = fortran_int_max(K, 1);
    // ldy = fortran_int_max(N, 1);
    ldx = fortran_int_max(M, 1);
    ldy = fortran_int_max(K, 1);
    ldz = fortran_int_max(M, 1);

    mem_buff = malloc(safe_M * safe_K * sizeof(fortran_doublereal)
                      + safe_K * safe_N * sizeof(fortran_doublereal)
                      + safe_M * safe_N * sizeof(fortran_doublereal));
    if (!mem_buff) {
        goto error;
    }
    a = mem_buff;
    b = a + safe_M * safe_K * sizeof(fortran_doublereal);
    c = b + safe_K * safe_N * sizeof(fortran_doublereal);

    params->TRANSX = 'N';
    params->TRANSY = 'N';
    params->A = &d_one;
    params->B = &d_zero;
    params->X = a;
    params->Y = b;
    params->Z = c;
    params->M = M;
    params->N = N;
    params->K = K;
    params->LDX = ldx;
    params->LDY = ldy;
    params->LDZ = ldz;

    return 1;
 error:
    free(mem_buff);
    memset(params, 0, sizeof(*params));
    PyErr_NoMemory();

    return 0;
}

/* *********************************
* Deallocate buffer
************************************* */

static NPY_INLINE void
release_DOUBLE_matm(GEMM_PARAMS_t *params)
{
    /* memory block base is in X */
    free(params->X);
    memset(params, 0, sizeof(*params));
}

/* ***************************
* Inner GUfunc loop
****************************** */

static void
DOUBLE_matmul(char **args, npy_intp *dimensions, npy_intp *steps,
              void *NPY_UNUSED(func))
{
INIT_OUTER_LOOP_3

    npy_intp len_m = *dimensions++;  // dimensions of left
    npy_intp len_k = *dimensions++;  // dimensions of inner
    npy_intp len_n = *dimensions++;  // dimensions of right
    npy_intp stride_x_m = *steps++;  // 1st arg
    npy_intp stride_x_k = *steps++;
    npy_intp stride_y_k = *steps++;  // 2nd arg
    npy_intp stride_y_n = *steps++;
    npy_intp stride_z_m = *steps++;  // output
    npy_intp stride_z_n = *steps++;
    GEMM_PARAMS_t params;
    LINEARIZE_DATA_t x_in, y_in, z_out;

    init_linearize_data(&x_in, len_k, len_m, stride_x_k, stride_x_m);
    init_linearize_data(&y_in, len_n, len_k, stride_y_n, stride_y_k);
    init_linearize_data(&z_out, len_n, len_m, stride_z_n, stride_z_m);

    if(init_DOUBLE_matm(&params, len_m, len_n, len_k)) {
        BEGIN_OUTER_LOOP_3
            linearize_DOUBLE_matrix(params.X, args[0], &x_in);
            linearize_DOUBLE_matrix(params.Y, args[1], &y_in);
            call_dgemm(&params);
            delinearize_DOUBLE_matrix(args[2], params.Z, &z_out);
        END_OUTER_LOOP
        release_DOUBLE_matm(&params);
    }
}

/*
*****************************************************************************
**                             UFUNC DEFINITION                            **
*****************************************************************************
*/

GUFUNC_FUNC_ARRAY_REAL(pdist_ratio);
GUFUNC_FUNC_ARRAY_REAL(cdist_ratio);
GUFUNC_FUNC_ARRAY_REAL(matmul);
GUFUNC_FUNC_ARRAY_REAL(norm);

GUFUNC_DESCRIPTOR_t gufunc_descriptors[] = {
    {"pdist_ratio", "(d,m),(d,n)->(),()", pdist_ratio__doc__,
     1, 2, 2, FUNC_ARRAY_NAME(pdist_ratio), ufn_types_1_4},
    {"cdist_ratio", "(d1,m),(d2,m),(d1,n),(d2,n)->(),()", cdist_ratio__doc__,
     1, 4, 2, FUNC_ARRAY_NAME(cdist_ratio), ufn_types_1_6},
    {"matmul", "(m,n),(n,p)->(m,p)", matmul__doc__,
     1, 2, 1, FUNC_ARRAY_NAME(matmul), ufn_types_1_3},
    {"norm", "(n)->()", norm__doc__,
     1, 1, 1, FUNC_ARRAY_NAME(norm), ufn_types_1_2}
};

/*
*****************************************************************************
**               Module initialization stuff                               **
*****************************************************************************
*/

static PyMethodDef GUfuncs_BLAS_Methods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_gufuncs_blas",
        NULL,
        -1,
        GUfuncs_BLAS_Methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyObject *PyInit__gufuncs_blas(void)
{
    PyObject *m;
    PyObject *d;
    PyObject *version;
    int failure;

    init_constants();
    m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    import_array();
    import_ufunc();

    d = PyModule_GetDict(m);

    version = PyString_FromString(gufuncs_blas_version_string);
    PyDict_SetItemString(d, "__version__", version);
    Py_DECREF(version);

    /* Load the ufunc operators into the module's namespace */
    failure = addUfuncs(d, gufunc_descriptors, 4);

    if (PyErr_Occurred() || failure) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load _gufuncs_blas module.");
        return NULL;
    }

    return m;
}
