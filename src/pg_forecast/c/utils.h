#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include <catalog/pg_type.h>

#ifndef UTILS_H
#define UTILS_H

double* pg_array_to_c_double(ArrayType* inp_arr, int* out_n, bool accept_nulls,
                            const char* caller);

#endif