#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/builtins.h" 
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "utils.h"
#include "arima.h"
#include <nlopt.h> 
#include <string.h>

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(arima_difference);
PG_FUNCTION_INFO_V1(arima_integrate);
PG_FUNCTION_INFO_V1(css_loss);
PG_FUNCTION_INFO_V1(arima_optimise);
PG_FUNCTION_INFO_V1(arima_forecast);

/* ------------------------------------------------------------------------- 
 * Differencing / Integration
 * ------------------------------------------------------------------------- */

PGDLLEXPORT Datum
arima_difference(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *vals_arr = PG_GETARG_ARRAYTYPE_P(0);
    int32 d = PG_GETARG_INT32(1);

    // No differencing required
    if (d <= 0)
    {
        PG_RETURN_ARRAYTYPE_P(vals_arr);
    }

    /* Convert psql array to C array */
    int n_vals;
    double* vals = get_1d_double_array(vals_arr, &n_vals, "arima_difference");

    /* Perform differencing */
    double *differenced_vals = arima_compute_difference(vals, n_vals, d);

    /* Convert to PG return type */
    ArrayType *pg_differenced = build_float8_array(differenced_vals, n_vals - d);
    PG_RETURN_ARRAYTYPE_P(pg_differenced);
}

PGDLLEXPORT Datum
arima_integrate(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *diff_arr = PG_GETARG_ARRAYTYPE_P(0);
    int32 d = PG_GETARG_INT32(1);
    ArrayType *initial_arr = PG_GETARG_ARRAYTYPE_P(2);

    // No differencing required
    if (d <= 0)
    {
        PG_RETURN_ARRAYTYPE_P(diff_arr);
    }

    /* Convert psql array to C array */
    int n_diff, n_initial;
    double* differences = get_1d_double_array(diff_arr, &n_diff, "arima_integrate");
    double* initial_vals = get_1d_double_array(initial_arr, &n_initial, "arima_integrate");

    if (n_initial != d)
    {
        ereport(ERROR, (errmsg("arima_integrate: number of initial vals must match differencing order")));
    }

    /* Perform integration */
    double *integrated = arima_compute_integration(differences, n_diff, d, initial_vals);

    /* Convert to PG return type */
    ArrayType *pg_integrated = build_float8_array(integrated, n_diff + d);
    PG_RETURN_ARRAYTYPE_P(pg_integrated);
}

/* ------------------------------------------------------------------------- 
 * CSS
 * ------------------------------------------------------------------------- */

PGDLLEXPORT Datum
css_loss(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *vals_arr  = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *phi_arr   = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType *theta_arr = PG_GETARG_ARRAYTYPE_P(2);
    int32 p = PG_GETARG_INT32(3);
    int32 q = PG_GETARG_INT32(4);
    float8 c = PG_GETARG_FLOAT8(5);

    /* Convert psql array to C array */
    int n_vals, n_phi, n_theta;
    double* vals = get_1d_double_array(vals_arr, &n_vals, "css_loss");
    double* phi = get_1d_double_array(phi_arr, &n_phi, "css_loss");
    double* theta = get_1d_double_array(theta_arr, &n_theta, "css_loss");
    
    check_arima_dims(p, q, n_phi, n_theta, "css_loss");

    /* CSS logic */
    double* resid = palloc(n_vals * sizeof(double));
    PG_RETURN_FLOAT8(css(vals, phi, theta, p, q, false, c, n_vals, NULL, resid));
}

/* ------------------------------------------------------------------------- 
 * Optimisation
 * ------------------------------------------------------------------------- */

PGDLLEXPORT Datum
arima_optimise(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *vals_arr  = PG_GETARG_ARRAYTYPE_P(0);
    int32 p = PG_GETARG_INT32(1);
    int32 q = PG_GETARG_INT32(2);
    bool include_c = PG_GETARG_BOOL(3);
    char* arima_method = text_to_cstring(PG_GETARG_TEXT_P(4));

    if (p < 0 || q < 0)
    {
        ereport(ERROR, 
                (errmsg("p and q must be positive integers")));
    }

    /* Convert psql array to C array */
    int n_vals;
    double* vals = get_1d_double_array(vals_arr, &n_vals, "arima_optimise");

    /* Call optimiser */
    double (*opt_objective)(unsigned, const double *, double *, void *);
    nlopt_algorithm opt_algorithm;
    opt_result_t opt_result;

    if (strcmp(arima_method, "Nelder-Mead") == 0)
    {
        opt_objective = _arima_objective_no_grad;
        opt_algorithm = NLOPT_LN_NELDERMEAD;
    }
    else if (strcmp(arima_method, "L-BFGS") == 0)
    {
        opt_objective = _arima_objective_grad;
        opt_algorithm = NLOPT_LD_LBFGS;
    }
    else if (strcmp(arima_method, "SLSQP") == 0)
    {
        opt_objective = _arima_objective_grad;
        opt_algorithm = NLOPT_LD_SLSQP;
    }
    else if (strcmp(arima_method, "MMA") == 0)
    {
        opt_objective = _arima_objective_grad;
        opt_algorithm = NLOPT_LD_MMA;
    }
    else if (strcmp(arima_method, "Subplex") == 0)
    {
        opt_objective = _arima_objective_grad;
        opt_algorithm = NLOPT_LN_SBPLX;
    }
    else
    {
        elog(ERROR, "Invalid optimiser provided: %s", arima_method);
    }
    opt_result = _arima_nlopt(vals, n_vals, p, q, include_c, opt_algorithm, arima_method, opt_objective);

    double c = include_c ? opt_result.params[p + q] : 0.0;

    /* Convert to arima_optimise_result return type */
    TupleDesc tupdesc;
    Datum values[5];
    bool nulls[5] = {false, false, false, false, false};
    HeapTuple tuple;
    Datum result;
    Oid type_oid = get_call_result_type(fcinfo, NULL, &tupdesc);
    if (!OidIsValid(type_oid))
    {
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_OBJECT),
                 errmsg("type \"arima_optimise_result\" does not exist")));
    }
    
    // Phi array
    values[0] = PointerGetDatum(build_float8_array(opt_result.params, p));

    // Theta array
    values[1] = PointerGetDatum(build_float8_array(opt_result.params + p, q));

    // Constant term
    values[2] = Float8GetDatum(c);

    // Residuals array
    double *last_q_resids = palloc(sizeof(double) * q);
    for (int i = 0; i < q; i++)
    {
        last_q_resids[i] = opt_result.resid[n_vals - i - 1];
    }
    values[3] = PointerGetDatum(build_float8_array(last_q_resids, q));

    // CSS
    values[4] = Float8GetDatum(opt_result.css);

    tuple = heap_form_tuple(tupdesc, values, nulls);
    result = HeapTupleGetDatum(tuple);
    ReleaseTupleDesc(tupdesc);
    PG_RETURN_DATUM(result);
}

/* ------------------------------------------------------------------------- 
 * Forecasting
 * ------------------------------------------------------------------------- */

PGDLLEXPORT Datum
arima_forecast(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *vals_arr  = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *resid_arr  = PG_GETARG_ARRAYTYPE_P(1);
    int32 p = PG_GETARG_INT32(2);
    int32 q = PG_GETARG_INT32(3);
    float8 c = PG_GETARG_FLOAT8(4);
    ArrayType *phi_arr  = PG_GETARG_ARRAYTYPE_P(5);
    ArrayType *theta_arr  = PG_GETARG_ARRAYTYPE_P(6);
    int32 horizon = PG_GETARG_INT32(7);

    if (p < 0 || q < 0 || horizon < 1)
    {
        ereport(ERROR, 
                (errmsg("p, q and horizon must be positive integers")));
    }

    /* Convert psql arrays to C arrays */
    int n_vals, n_resid, n_phi, n_theta;
    double *vals = get_1d_double_array(vals_arr, &n_vals, "arima_forecast");
    double *resid = get_1d_double_array(resid_arr, &n_resid, "arima_forecast");
    double *phi = get_1d_double_array(phi_arr, &n_phi, "arima_forecast");
    double *theta = get_1d_double_array(theta_arr, &n_theta, "arima_forecast");

    /* Predict */
    double* yhat = _arima_predict(vals, n_vals, resid, p, q, c, phi, theta, horizon);

    /* Convert to PG return type */
    ArrayType *pg_yhat = build_float8_array(yhat, horizon);
    PG_RETURN_ARRAYTYPE_P(pg_yhat);
}