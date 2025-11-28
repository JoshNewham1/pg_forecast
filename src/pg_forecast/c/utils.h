#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include <catalog/pg_type.h>

#ifndef UTILS_H
#define UTILS_H

/* PG API helpers */
double* pg_array_to_c_double(ArrayType* inp_arr, int* out_n, bool accept_nulls,
                            const char* caller);

/* Maths helpers */
double arr_mean(double *x, int n);
#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
    
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a <= _b ? _a : _b; })

#ifndef M_PI
    #define M_PI (3.14159265358979323846)
#endif

#endif