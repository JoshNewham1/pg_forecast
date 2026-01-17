#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include <catalog/pg_type.h>

#ifndef UTILS_H
#define UTILS_H

/* PG API helpers */
double* pg_array_to_c_double(ArrayType* inp_arr, int* out_n, bool accept_nulls,
                            const char* caller);

double *get_1d_double_array(ArrayType *arr, int *n, const char *fn);

void check_arima_dims(int p, int q, int n_phi, int n_theta, const char *fn);

ArrayType* build_float8_array(double* arr, int size);

/* Maths helpers */
double arr_mean(double *x, int n);
#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))

#ifndef M_PI
    #define M_PI (3.14159265358979323846)
#endif

#endif
