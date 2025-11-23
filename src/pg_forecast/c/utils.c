#include "utils.h"

double* pg_array_to_c_double(ArrayType* inp_arr, int* out_n, bool accept_nulls,
                            const char* caller)
{
    Datum* arr_d;
    bool* arr_nulls;
    deconstruct_array(inp_arr,
                      FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd',
                      &arr_d, &arr_nulls, out_n);
    
    double* out_arr = palloc(*out_n * sizeof(double));
    for (int i = 0; i < *out_n; i++)
    {
        out_arr[i] = DatumGetFloat8(arr_d[i]);
        if (!accept_nulls && arr_nulls[i])
            ereport(ERROR, (errmsg("%s: input array contains NULLs", caller)));
    }
    return out_arr;
}