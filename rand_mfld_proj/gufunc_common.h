/* Common code for creating GUFuncs
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

/*       Table of Contents
49.  Includes
65.  Outer loop macros
126. Error signaling functions
151. Constants
179. Ufunc definition
*/

#ifndef GUC_INCLUDE
#define GUC_INCLUDE

/*
*****************************************************************************
**                             INCLUDES                                    **
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

/*
*****************************************************************************
**                         OUTER LOOP MACROS                               **
*****************************************************************************
*/

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

#define INIT_OUTER_LOOP_4       \
 INIT_OUTER_LOOP_3           \
 npy_intp s3 = *steps++;

#define INIT_OUTER_LOOP_5 \
 INIT_OUTER_LOOP_4\
 npy_intp s4 = *steps++;

#define INIT_OUTER_LOOP_6  \
 INIT_OUTER_LOOP_5\
 npy_intp s5 = *steps++;

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
    d_one  = 1.0;
    d_zero = 0.0;
    d_minus_one = -1.0;
    d_inf = NPY_INFINITY;
    d_nan = NPY_NAN;
    d_eps = npy_spacing(d_one);
}

/*
*****************************************************************************
**                             UFUNC DEFINITION                            **
*****************************************************************************
*/
static void *null_data_4[] = { (void *)NULL, (void *)NULL, (void *)NULL, (void *)NULL };

static char ufn_types_1_2[] = { NPY_DOUBLE, NPY_DOUBLE };
static char ufn_types_1_3[] = { NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE };
static char ufn_types_1_4[] = { NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE };
static char ufn_types_1_5[] = { NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE };
static char ufn_types_1_6[] = { NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE };

#define FUNC_ARRAY_NAME(NAME) NAME ## _funcs

#define GUFUNC_FUNC_ARRAY_REAL(NAME)                    \
    static PyUFuncGenericFunction                       \
    FUNC_ARRAY_NAME(NAME)[] = {                         \
        DOUBLE_ ## NAME                                 \
    }

typedef struct gufunc_descriptor_struct {
    char *name;
    char *signature;
    char *doc;
    int ntypes;
    int nin;
    int nout;
    PyUFuncGenericFunction *funcs;
    char *types;
} GUFUNC_DESCRIPTOR_t;

static int
addUfuncs(PyObject *dictionary, const GUFUNC_DESCRIPTOR_t guf_descriptors[],
            const int gufunc_count) {
    PyObject *f;
    int i;
    for (i = 0; i < gufunc_count; i++) {
        const GUFUNC_DESCRIPTOR_t* d = &guf_descriptors[i];
        f = PyUFunc_FromFuncAndDataAndSignature(d->funcs, null_data_4, d->types,
                                                d->ntypes, d->nin, d->nout, PyUFunc_None,
                                                d->name, d->doc, 0, d->signature);
        if (f == NULL) {
            return -1;
        }
        PyDict_SetItemString(dictionary, d->name, f);
        Py_DECREF(f);
    }
    return 0;
}

#endif
