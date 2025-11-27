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

#endif