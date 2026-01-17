#include "utils.h"
#include <string.h>
#include <math.h>

/* PG API helpers */
double* pg_array_to_c_double(ArrayType* inp_arr, int* out_n, bool accept_nulls,
                            const char* caller)
{
    if (ARR_ELEMTYPE(inp_arr) != FLOAT8OID)
        ereport(ERROR, (errmsg("%s: array must be float8[]", caller)));

    *out_n = ArrayGetNItems(ARR_NDIM(inp_arr), ARR_DIMS(inp_arr));

    // Fast path: No NULLs, recast pointer
    if (!ARR_HASNULL(inp_arr))
    {
        return (double*)ARR_DATA_PTR(inp_arr);
    }

    Datum* arr_d;
    bool* arr_nulls;
    int n;
    deconstruct_array(inp_arr,
                      FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd',
                      &arr_d, &arr_nulls, &n);
    
    double* out_arr = palloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        if (arr_nulls[i])
        {
            if (!accept_nulls)
                ereport(ERROR, (errmsg("%s: input array contains NULLs", caller)));
            out_arr[i] = NAN;
        }
        else
        {
            out_arr[i] = DatumGetFloat8(arr_d[i]);
        }
    }
    return out_arr;
}

double *
get_1d_double_array(ArrayType *arr, int *n, const char *caller)
{
    if (ARR_NDIM(arr) > 1)
        ereport(ERROR, (errmsg("%s: array must be 1-D", caller)));

    return pg_array_to_c_double(arr, n, false, caller);
}

void
check_arima_dims(int p, int q, int n_phi, int n_theta, const char *caller)
{
    if (n_phi != p)
        ereport(ERROR, (errmsg("%s: phi must have %d elements", caller, p)));
    if (n_theta != q)
        ereport(ERROR, (errmsg("%s: theta must have %d elements", caller, q)));
}

ArrayType* build_float8_array(double* arr, int size)
{
    Datum *darray = palloc(sizeof(Datum) * size);
    for (int i = 0; i < size; i++)
    {
        darray[i] = Float8GetDatum(arr[i]);
    }
    ArrayType *pg_arr = construct_array(darray, size, FLOAT8OID,
                                        sizeof(double), FLOAT8PASSBYVAL,
                                        'd');
    return pg_arr;
}

/* Maths helpers */
double arr_mean(double *x, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += x[i];
    return sum / n;
}