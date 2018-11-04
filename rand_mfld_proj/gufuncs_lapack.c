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
53.   Includes
73.   Docstrings
143.  BLAS/Lapack calling functions
185.  Data rearrangement functions
360.  QR
577.  SOLVE
710.  EIGVALS
867.  SINGVALS
1042. Ufunc definition
1064. Module initialization stuff
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
"    Matrix with orthonormal columns.\n\n"
"Omits\n-----\n"
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

PyDoc_STRVAR(eigvalsh__doc__,
//"eigvalsh(A: ndarray) -> (L: ndarray)\n\n"
"Eigenvalues of hermitian matrix.\n\n"
"Find the `lambda` such that `A x == lambda x` for some `x`.\n\n"
"Parameters\n-----------\n"
"A: ndarray (...,N,N)\n"
"    Matrix whose eigenvalues we compute.\n"
"Returns\n-------\n"
"L: ndarray (...,N)\n"
"    Vector of eigenvalues.\n");

PyDoc_STRVAR(singvals__doc__,
//"singvals(A: ndarray) -> (S: ndarray)\n\n"
"Singular values of matrix.\n\n"
"Find the `s` such that `A v == s u, A' u == s v,` for some `u,v`.\n\n"
"Parameters\n-----------\n"
"A: ndarray (...,M,N)\n"
"    Matrix whose singular values we compute.\n"
"Returns\n-------\n"
"S: ndarray (...,K)\n"
"    Vector of singular values. `K = min(M,N)`\n");

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

/* qr decomposition of a */
/* a -> r, v, tau */
extern int
FNAME(dgeqrf)(int *m, int *n, double *a, int *lda, double *tau,
              double *work, int * lwork, int *info);

/* v, tau -> q */
extern int
FNAME(dorgqr)(int *m, int *n, int *k,
             double *a, int *lda, double *tau,
             double *work, int * lwork, int *info);

/* solve a x = b for x */
extern int
FNAME(dgesv)(int *n, int *nrhs,
             double *a, int *lda, int * ipiv,
             double *b, int *ldb, int *info);

/* eigenvalue decomposition */
extern int
FNAME(dgeev)(char *jobvl, char *jobvr, int *n,
             double *a, int *lda, double *wr, double *wi,
             double *vl, int *ldvl, double *vr, int *ldvr,
             double *work, int *lwork, int *info);

/* singular value decomposition */
extern int
FNAME(dgesdd)(char *jobz, int *m, int *n,
             double *a, int *lda, double *s,
             double *u, int *ldu, double *v, int *ldv,
             double *work, int *lwork, int *iwork, int *info);

/*
*****************************************************************************
**                    DATA REARRANGEMENT FUNCTIONS                         **
*****************************************************************************
*/

              /* rearranging of 2D matrices using blas */

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

// char *qr_m_signature = "(m,n)->(m,m),(m,n)";  // m<n
// char *qr_n_signature = "(m,n)->(m,n),(n,n)";  // m>n

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
init_DOUBLE_qr(GEQRF_PARAMS_t *params, npy_intp M_in, npy_intp N_in, npy_intp NC_in)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *b, *c, *d;
    fortran_int M = (fortran_int)M_in;
    fortran_int N = (fortran_int)N_in;
    fortran_int NC = (fortran_int)NC_in;
    size_t safe_M = M_in;
    size_t safe_N = N_in;
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
do_DOUBLE_qr(const void *A, void *Q, GEQRF_PARAMS_t *params,
             const LINEARIZE_DATA_t *a_in,  const LINEARIZE_DATA_t *q_out)
{
    // copy input to buffer
    linearize_DOUBLE_matrix(params->A, A, a_in);
    // QR decompose
    call_dgeqrf(params);
    if (params->INFO < 0) {
      return -1;
    }
    // Build Q
    call_dorgqr(params);
    if (params->INFO < 0) {
      return -1;
    }
    // Copy Q from buffer
    delinearize_DOUBLE_matrix(Q, params->A, q_out);
    return 0;
}

static void
DOUBLE_qr(char **args, npy_intp *dimensions, npy_intp *steps, int complete)
{
INIT_OUTER_LOOP_2
    npy_intp len_m = *dimensions++;  // rows
    npy_intp len_n = *dimensions++;  // columns
    npy_intp stride_a_m = *steps++;  // rows
    npy_intp stride_a_n = *steps++;
    npy_intp stride_q_m = *steps++;  // rows
    npy_intp stride_q_k = *steps++;
    int error_occurred = get_fp_invalid_and_clear();
    GEQRF_PARAMS_t params;
    LINEARIZE_DATA_t a_in, q_out;
    npy_intp len_nc = complete ? len_m : len_n;

    if(len_m < len_nc) {//signature demands a wide matrix for q, which is impossible for qr_n.
        // PyErr_SetString(PyExc_ValueError, "qr_n can only be called when m >= n.");
        error_occurred = 1;
        init_linearize_data(&q_out, len_nc, len_m, stride_q_k, stride_q_m);
        nan_DOUBLE_matrix(args[1], &q_out);
    } else {
        if(init_DOUBLE_qr(&params, len_m, len_n, len_nc)){
            init_linearize_data(&a_in, len_n, len_m, stride_a_n, stride_a_m);
            init_linearize_data(&q_out, len_nc, len_m, stride_q_k, stride_q_m);
            BEGIN_OUTER_LOOP_2
                int not_ok;
                not_ok = do_DOUBLE_qr(args[0], args[1], &params, &a_in, &q_out);
                if (not_ok) {
                    error_occurred = 1;
                    nan_DOUBLE_matrix(args[1], &q_out);
                }
            END_OUTER_LOOP_2
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

// char *solve_signature = "(n,n),(n,nrhs)->(n,nrhs)";

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
init_dgesv(GESV_PARAMS_t *params, npy_intp N_in, npy_intp NRHS_in)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *a, *b, *c;
    fortran_int N = (fortran_int)N_in;
    fortran_int NRHS = (fortran_int)NRHS_in;
    size_t safe_N = N_in;
    size_t safe_NRHS = NRHS_in;
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

    if(init_dgesv(&params, len_n, len_nrhs)){
        init_linearize_data(&a_in, len_n, len_n, stride_a_c, stride_a_r);
        init_linearize_data(&b_in, len_nrhs, len_n, stride_b_c, stride_b_r);
        init_linearize_data(&x_out, len_nrhs, len_n, stride_x_c, stride_x_r);
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
        END_OUTER_LOOP_3
        release_dgesv(&params);
    }
    set_fp_invalid_or_clear(error_occurred);
}

/*
******************************************************************************
**                              EIGVALSH                                    **
******************************************************************************
*/

// char *eigvalsh_signature = "(n,n)->(n)";

typedef struct geev_params_struct
{
    void *A; /* A is (N,N) of base type */
    void *VL; /* B is (N,N) of base type */
    void *VR; /* B is (N,N) of base type */
    void *WR; /* W is (N,) of base type, work for _geqrf */
    void *WI; /* W is (N,) of base type, work for _geqrf */
    void *W; /* W is (LW,) of base type, work for _geqrf */

    fortran_int N;
    fortran_int LDA;
    fortran_int LDVL;
    fortran_int LDVR;
    fortran_int LW;
    fortran_int INFO;
    char JOBVL;
    char JOBVR;
} GEEV_PARAMS_t;

/* ************************************************
* Calling BLAS/Lapack functions _syevd
*************************************************** */

static NPY_INLINE void
call_dgeev(GEEV_PARAMS_t *params)
{
    // A,B are modified by ?GELS to carry LU info & X
    LAPACK(dgeev)(&params->JOBVL, &params->JOBVR, &params->N,
                params->A, &params->LDA, params->WR, params->WI,
                params->VL, &params->LDVL, params->VR, &params->LDVR,
                params->W, &params->LW, &params->INFO);
}

/* *************************************************************************
* Initialize the parameters to use in the lapack functions _syevd
* Handles buffer allocation
************************************************************************** */
static NPY_INLINE int
init_dgeev(GEEV_PARAMS_t *params, npy_intp N_in)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *b, *c, *d;
    fortran_int N = (fortran_int)N_in;
    size_t safe_N = N_in;
    fortran_int lda = fortran_int_max(N, 1);
    fortran_doublereal work_size;
    mem_buff = malloc(safe_N * safe_N * sizeof(fortran_doublereal)
                    + safe_N * sizeof(fortran_doublereal)
                    + safe_N * sizeof(fortran_doublereal));
    if (!mem_buff) {
        goto error;
    }
    a = mem_buff;
    b = a + safe_N * safe_N * sizeof(fortran_doublereal);
    c = b + safe_N * sizeof(fortran_doublereal);

    params->A = a;
    params->WR = b;
    params->WR = c;
    params->W = &work_size;
    params->VL = NULL;
    params->VR = NULL;
    params->N = N;
    params->LDA = lda;
    params->LDVL = lda;
    params->LDVR = lda;
    params->LW = -1;
    params->INFO = 0;
    params->JOBVL = 'N';
    params->JOBVR = 'N';

    call_dsyevd(params);
    if (params->INFO) {
        goto error;
    }
    fortran_int LW = (fortran_int)work_size;
    size_t safe_LW = LW;

    mem_buff2 = malloc(safe_LW * sizeof(fortran_doublereal));
    if (!mem_buff2) {
        goto error;
    }
    d = mem_buff2;

    params->W = d;
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
release_dgeev(GEEV_PARAMS_t *params)
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
    GEEV_PARAMS_t params;
    LINEARIZE_DATA_t a_in;
    LINEARIZE_VDATA_t e_out;

    if(init_dgeev(&params, len_n)){
        init_linearize_data(&a_in, len_n, len_n, stride_a_c, stride_a_r);
        init_linearize_vdata(&e_out, len_n, stride_e);
        BEGIN_OUTER_LOOP_2
            int not_ok;
            linearize_DOUBLE_matrix(params.A, args[0], &a_in);
            call_dgeev(&params);
            not_ok = params.INFO;
            if (not_ok) {
                error_occurred = 1;
                nan_DOUBLE_vec(args[1], &e_out);
            } else {
                delinearize_DOUBLE_vec(args[1], params.WR, &e_out);
            }
        END_OUTER_LOOP_2
        release_dgeev(&params);
    }
    set_fp_invalid_or_clear(error_occurred);
}

/*
******************************************************************************
**                              SINGVALS                                    **
******************************************************************************
*/

// char *singvals_m_signature = "(m,n)->(m)";
// char *singvals_n_signature = "(m,n)->(n)";

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
init_dgesdd(GESDD_PARAMS_t *params, npy_intp M_in, npy_intp N_in)
{
    npy_uint8 *mem_buff = NULL;
    npy_uint8 *mem_buff2 = NULL;
    npy_uint8 *a, *b, *c, *d;
    fortran_int M = (fortran_int)M_in;
    fortran_int N = (fortran_int)N_in;
    size_t safe_M = M_in;
    size_t safe_N = N_in;
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

    if(init_dgesdd(&params, len_m, len_n)){
        init_linearize_data(&a_in, len_n, len_m, stride_a_c, stride_a_r);
        init_linearize_vdata(&s_out, len_k, stride_s);
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
        END_OUTER_LOOP_2
        release_dgesdd(&params);
    }
    set_fp_invalid_or_clear(error_occurred);
}



/*
*****************************************************************************
**                             UFUNC DEFINITION                            **
*****************************************************************************
*/

GUFUNC_FUNC_ARRAY_REAL(qr);
GUFUNC_FUNC_ARRAY_REAL(solve);
GUFUNC_FUNC_ARRAY_REAL(eigvalsh);
GUFUNC_FUNC_ARRAY_REAL(singvals);

GUFUNC_DESCRIPTOR_t gufunc_descriptors[] = {
    {"qr", "(m,n)->(m,n)", qr__doc__,
     1, 1, 1, FUNC_ARRAY_NAME(qr), ufn_types_1_2},
    {"solve", "(n,n),(n,nrhs)->(n,nrhs)", solve__doc__,
     1, 2, 1, FUNC_ARRAY_NAME(solve), ufn_types_1_3},
    {"eigvalsh", "(n,n)->(n)", eigvalsh__doc__,
     1, 1, 1, FUNC_ARRAY_NAME(eigvalsh), ufn_types_1_2},
    {"singvals", "(m,n)->(n)", singvals__doc__,
     1, 1, 1, FUNC_ARRAY_NAME(singvals), ufn_types_1_2}
};

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
    int failure;

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
    failure = addUfuncs(d, gufunc_descriptors, 4);

    if (PyErr_Occurred() || failure) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot load _gufuncs_lapack module.");
        return NULL;
    }

    return m;
}
