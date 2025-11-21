#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "funcapi.h"

#include <stdio.h>
#include <math.h>
#include "arima.h"
#include <nlopt.h>

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
    
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a <= _b ? _a : _b; })

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(css_loss);
PG_FUNCTION_INFO_V1(optimise_arima);

/* ARIMA loss function */
/* 
  Calculates conditional sum of squares (CSS) and optionally the gradient
  If the gradient isn't required, leave `grad` = NULL
  If the gradient is required, leave enough space for p+q values
*/
double css(double *vals, double *phi, double *theta, int p, int q, int n_vals, double* grad, double* resid)
{
    int ncond = max(p, q);
    memset(resid, 0, n_vals * sizeof(double));
    double ssq = 0.0;
    double **dphi, **dtheta = NULL;

    if (grad != NULL)
    {
        memset(grad, 0, (p + q) * sizeof(double));
        dphi = palloc(sizeof(double*) * p);
        for (int k = 0; k < p; k++)
        {
            dphi[k] = palloc0(sizeof(double) * n_vals);
        }
        dtheta = palloc(sizeof(double*) * q);
        for (int k = 0; k < q; k++)
        {
            dtheta[k] = palloc0(sizeof(double) * n_vals);
        }
    }

    for (int t = ncond; t < n_vals; t++)
    {
        double tmp = vals[t];

        // AR part: subtract phi_j * y_{t-j-1}
        for (int j = 0; j < p; j++)
        {
            tmp -= phi[j] * vals[t-j-1];
        }

        // MA part: subtract theta_j * resid_{t-j-1}
        for (int j = 0; j < min(t-ncond, q); j++)
        {
            tmp -= theta[j] * resid[t-j-1];
        }

        resid[t] = tmp;
        if (!isnan(tmp) && isfinite(tmp))
        {
            ssq += tmp * tmp;
        }

        if (grad != NULL)
        {
            /* Compute derivatives d e_t / d phi_k and d e_t / d theta_k recursively */
            int max_ma = min(t - ncond, q);
            // phi derivatives: d e_t / d phi_k = - y_{t-k-1} - sum_j theta_j * d e_{t-j-1} / d phi_k
            for (int k = 0; k < p; k++)
            {
                double d = 0.0;
                int idx = t - k - 1;
                if (idx >= 0) d = - vals[idx];
                // Add MA contribution (recursive)
                for (int j = 0; j < max_ma; j++)
                {
                    d -= theta[j] * dphi[k][t-j-1];
                }
                dphi[k][t] = d;
                // Accumulate gradient: d(CSS)/dphi_k = 2 * sum_t e_t * d e_t / dphi_k
                if (isfinite(tmp) && isfinite(d))
                {
                    grad[k] += 2.0 * tmp * dphi[k][t];
                }
            }

            // theta derivatives: d e_t / d theta_k = - e_{t-k-1} - sum_j theta_j * d e_{t-j-1} / d theta_k */
            for (int k = 0; k < q; k++)
            {
                double d = 0.0;
                int idx = t - k - 1;
                if (idx >= 0) d = - resid[idx];
                for (int j = 0; j < max_ma; j++)
                {
                    d -= theta[j] * dtheta[k][t-j-1];
                }
                dtheta[k][t] = d;
                if (isfinite(tmp) && isfinite(d))
                {
                    grad[p + k] += 2.0 * tmp * dtheta[k][t];
                }
            }
        }
    }
    // Free derivative arrays
    if (grad != NULL)
    {
        for (int k = 0; k < p; k++) pfree(dphi[k]);
        if (p>0) pfree(dphi);
        for (int k = 0; k < q; k++) pfree(dtheta[k]);
        if (q>0) pfree(dtheta);
    }
    return ssq;
}

Datum
css_loss(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *vals_arr  = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *phi_arr   = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType *theta_arr = PG_GETARG_ARRAYTYPE_P(2);
    int32 p = PG_GETARG_INT32(3);
    int32 q = PG_GETARG_INT32(4);

    /* Validate arguments */
    if (ARR_NDIM(vals_arr) != 1 || ARR_NDIM(phi_arr) != 1 || ARR_NDIM(theta_arr) != 1)
    {
        ereport(ERROR,
                (errmsg("vals, phi and theta must be 1-D arrays")));
    }

    int n_vals  = ArrayGetNItems(ARR_NDIM(vals_arr), ARR_DIMS(vals_arr));
    int n_phi   = ArrayGetNItems(ARR_NDIM(phi_arr), ARR_DIMS(phi_arr));
    int n_theta = ArrayGetNItems(ARR_NDIM(theta_arr), ARR_DIMS(theta_arr));
    if (n_phi != p)
    {
        ereport(ERROR,
                (errmsg("phi array must have p elements")));
    }
    if (n_theta != q)
    {
        ereport(ERROR,
                (errmsg("theta array must have q elements")));
    }

    /* Convert psql array to C array */
    Datum *vals_d, *phi_d, *theta_d;
    bool *vals_nulls, *phi_nulls, *theta_nulls;

    deconstruct_array(vals_arr,
                      FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd',
                      &vals_d, &vals_nulls, &n_vals);

    deconstruct_array(phi_arr,
                      FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd',
                      &phi_d, &phi_nulls, &n_phi);

    deconstruct_array(theta_arr,
                      FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd',
                      &theta_d, &theta_nulls, &n_theta);
    double *vals, *phi, *theta;
    vals  = palloc(sizeof(double) * n_vals);
    phi   = palloc(sizeof(double) * n_phi);
    theta = palloc(sizeof(double) * n_theta);

    for (int i = 0; i < n_vals; i++)
        vals[i] = DatumGetFloat8(vals_d[i]);

    for (int i = 0; i < n_phi; i++)
        phi[i] = DatumGetFloat8(phi_d[i]);

    for (int i = 0; i < n_theta; i++)
        theta[i] = DatumGetFloat8(theta_d[i]);
    
    /* CSS logic */
    double* resid = palloc(n_vals * sizeof(double));
    PG_RETURN_FLOAT8(css(vals, phi, theta, p, q, n_vals, NULL, resid));
}

/* ARIMA optimisation */
Datum
optimise_arima(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *vals_arr  = PG_GETARG_ARRAYTYPE_P(0);
    int32 p = PG_GETARG_INT32(1);
    int32 q = PG_GETARG_INT32(2);
    int n_vals = ArrayGetNItems(ARR_NDIM(vals_arr), ARR_DIMS(vals_arr));
    char* arima_method = text_to_cstring(PG_GETARG_TEXT_P(3));

    /* Validate arguments */
    if (ARR_NDIM(vals_arr) != 1)
    {
        ereport(ERROR,
                (errmsg("vals must be a 1-D array")));
    }
    if (p < 1 || q < 1)
    {
        ereport(ERROR,
                (errmsg("p and q must be positive integers")));
    }

    /* Convert psql array to C array */
    Datum *vals_d;
    bool *vals_nulls;

    deconstruct_array(vals_arr, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, 'd',
                      &vals_d, &vals_nulls, &n_vals);
    double *vals = palloc(sizeof(double) * n_vals);
    for (int i = 0; i < n_vals; i++)
        vals[i] = DatumGetFloat8(vals_d[i]);

    /* Call optimiser */
    opt_result_t opt_result;
    int n_params = p + q;
    if (strcmp(arima_method, "Nelder-Mead") == 0)
    {
        opt_result = optimise_arima_nelder(vals, n_vals, p, q);
    }
    else if (strcmp(arima_method, "L-BFGS") == 0)
    {
        opt_result = optimise_arima_lbfgs(vals, n_vals, p, q);
    }
    else
    {
        elog(ERROR, "Invalid optimiser provided: %s\n", arima_method);
    }

    /* Warn for risky bounds */
    for (int i = 0; i < n_params; i++)
    {
        double val = opt_result.params[i];
        if (val > 1.0 || val < -1.0)
        {
            elog(WARNING, "phi/theta coefficient outside of [-1, 1]: %f\n", val);
        }
    }

    /* Convert to optimise_arima_result return type */
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
                 errmsg("type \"optimise_arima_result\" does not exist")));
    }
    
    // Phi array
    Datum *phi_array = palloc(sizeof(Datum) * p);
    for (int i = 0; i < p; i++)
    {
        phi_array[i] = Float8GetDatum(opt_result.params[i]);
    }
    ArrayType *phi_converted = construct_array(phi_array, p, FLOAT8OID, sizeof(double), FLOAT8PASSBYVAL, 'd');
    values[0] = PointerGetDatum(phi_converted);
    nulls[0] = false;

    // Theta array
    Datum *theta_array = palloc(sizeof(Datum) * q);
    for (int i = 0; i < q; i++)
    {
        theta_array[i] = Float8GetDatum(opt_result.params[p+i]);
    }
    ArrayType *theta_converted = construct_array(theta_array, q, FLOAT8OID, sizeof(double), FLOAT8PASSBYVAL, 'd');
    values[1] = PointerGetDatum(theta_converted);
    nulls[1] = false;

    // Residuals array
    Datum *resid_array = palloc(sizeof(Datum) * q);
    for (int i = 0; i < q; i++)
    {
        resid_array[i] = Float8GetDatum(opt_result.resid[n_vals-i-1]);
    }
    ArrayType *resid_converted = construct_array(resid_array, q, FLOAT8OID, sizeof(double), FLOAT8PASSBYVAL, 'd');
    values[2] = PointerGetDatum(resid_converted);
    nulls[2] = false;

    tuple = heap_form_tuple(tupdesc, values, nulls);
    result = HeapTupleGetDatum(tuple);
    ReleaseTupleDesc(tupdesc);
    PG_RETURN_DATUM(result);
}

static double arima_nelder_objective(unsigned n, const double *x, double *grad, void *data)
{
    css_data_t *d = (css_data_t *)data;
    double *phi = (double *)x;
    double* theta = (double *)(x + d->p);
    double* resid = (double *)(d->resid);

    // Gradient is not required by Nelder-Mead algorithm
    return css(d->vals, phi, theta, d->p, d->q, d->n_vals, NULL, resid);
}

static double lbfgs_objective(unsigned n, const double *x, double *grad, void *data)
{
    css_data_t *d = (css_data_t *)data;
    double *phi = (double *)x;
    double* theta = (double *)(x + d->p);
    double* resid = (double *)(d->resid);
    
    return css(d->vals, phi, theta, d->p, d->q, d->n_vals, grad, resid);
}

opt_result_t optimise_arima_nelder(double *vals, int n_vals, int p, int q)
{
    int n_params = p + q;
    nlopt_opt opt = nlopt_create(NLOPT_LN_NELDERMEAD, n_params);
    double* resid = palloc(n_vals * sizeof(double));

    css_data_t data = {vals, n_vals, p, q, resid};
    nlopt_set_min_objective(opt, arima_nelder_objective, &data);

    // Lower and upper bounds (stationarity assumed to be enforced)
    double *lb = palloc(sizeof(double) * n_params);
    double *ub = palloc(sizeof(double) * n_params);
    for (int i = 0; i < n_params; i++)
    {
        lb[i] = -2.0;
        ub[i] = 2.0;
    }
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);

    // Initial guess - all parameters are 0.1
    double *x = palloc(sizeof(double) * n_params);
    for (int i = 0; i < n_params; i++) x[i] = 0.1;

    double result;
    if (nlopt_optimize(opt, x, &result) < 0)
    {
        ereport(ERROR,
        errmsg("NLopt Nelder-Mead failed\n"));
    }

    nlopt_destroy(opt);
    elog(INFO, "Min CSS: %f\n", result);
    opt_result_t return_val = {x, resid};
    return return_val;
}

opt_result_t optimise_arima_lbfgs(double *vals, int n_vals, int p, int q)
{
    int n_params = p + q;
    nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, n_params);
    double* resid = palloc(n_vals * sizeof(double));

    css_data_t data = {vals, n_vals, p, q, resid};
    nlopt_set_min_objective(opt, lbfgs_objective, &data);

    // Lower and upper bounds (stationarity assumed to be enforced)
    double *lb = palloc(sizeof(double) * n_params);
    double *ub = palloc(sizeof(double) * n_params);
    for (int i = 0; i < n_params; i++)
    {
        lb[i] = -2.0;
        ub[i] = 2.0;
    }
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);

    // Initial guess - all parameters are 0.1
    double *x = palloc(sizeof(double) * n_params);
    for (int i = 0; i < n_params; i++) x[i] = 0.1;

    double result;
    if (nlopt_optimize(opt, x, &result) < 0)
    {
        ereport(ERROR,
        errmsg("NLopt L-BFGS failed\n"));
    }

    nlopt_destroy(opt);
    elog(INFO, "Min CSS: %f\n", result);

    opt_result_t return_val = {x, resid};
    return return_val;
}