#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "funcapi.h"

#include "autoarima.h"
#include "utils.h"
#include "constants.h"


PG_FUNCTION_INFO_V1(kpss);
PG_FUNCTION_INFO_V1(aicc);

PGDLLEXPORT Datum
kpss(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *vals_arr = PG_GETARG_ARRAYTYPE_P(0);
    float8 p_value = PG_GETARG_FLOAT8(1);

    /* Validate arguments */
    if (ARR_NDIM(vals_arr) != 1)
    {
        ereport(ERROR,
                (errmsg("vals must be a 1-D array")));
    }
    double crit_val = 0.0;
    for (int i = 0; i < AUTOARIMA_NUM_P_VALUES; i++)
    {
        if (p_value == AUTOARIMA_P_VALUES[i])
            crit_val = AUTOARIMA_KPSS_CRITICAL_VALS[i];
    }
    if (crit_val == 0.0)
    {
        ereport(ERROR,
                (errmsg("p_value must be in supported list")));
    }

    /* Convert to C array*/
    int n_vals;
    double* vals = pg_array_to_c_double(vals_arr, &n_vals, false, "kpss");

    /* Calculate KPSS and test against critical value */
    double kpss_score = _kpss(vals, n_vals);
    bool test_passed = kpss_score <= crit_val;
    
    /* Convert to kpss_result return type */
    TupleDesc tupdesc;
    Datum values[3];
    bool nulls[3];
    HeapTuple tuple;
    Datum result;
    Oid type_oid = get_call_result_type(fcinfo, NULL, &tupdesc);
    if (!OidIsValid(type_oid))
    {
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_OBJECT),
                 errmsg("type \"kpss_result\" does not exist")));
    }

    values[0] = Float8GetDatum(kpss_score);
    nulls[0] = false;
    values[1] = Float8GetDatum(crit_val);
    nulls[1] = false;
    values[2] = BoolGetDatum(test_passed);
    nulls[2] = false;

    tuple = heap_form_tuple(tupdesc, values, nulls);
    result = HeapTupleGetDatum(tuple);
    ReleaseTupleDesc(tupdesc);
    PG_RETURN_DATUM(result);
}

PGDLLEXPORT Datum
aicc(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    float8 css = PG_GETARG_FLOAT8(0);
    int32 p = PG_GETARG_INT32(1);
    int32 q = PG_GETARG_INT32(2);
    int32 k = PG_GETARG_INT32(3);
    int32 n_vals = PG_GETARG_INT32(4);

    double aicc = arima_aicc(css, p, q, k, n_vals);

    PG_RETURN_FLOAT8(aicc);
}