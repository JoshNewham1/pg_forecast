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
#include "constants.h"
#include "utils.h"
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

PG_FUNCTION_INFO_V1(arima_difference);
PG_FUNCTION_INFO_V1(arima_integrate);
PG_FUNCTION_INFO_V1(css_loss);
PG_FUNCTION_INFO_V1(arima_optimise);
PG_FUNCTION_INFO_V1(arima_forecast);

/* Differencing and Integration */
Datum
arima_difference(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *vals_arr = PG_GETARG_ARRAYTYPE_P(0);
    int32 d = PG_GETARG_INT32(1);

    /* Validate arguments */
    if (ARR_NDIM(vals_arr) != 1)
    {
        ereport(ERROR,
                (errmsg("vals must be a 1-D array")));
    }
    // No differencing required
    if (d <= 0)
    {
        PG_RETURN_ARRAYTYPE_P(vals_arr);
    }

    /* Convert psql array to C array */
    int n_vals;
    double* vals = pg_array_to_c_double(vals_arr, &n_vals, false, "arima_difference");

    /* Perform differencing */
    int vals_len = n_vals;
    double *next = palloc0(vals_len * sizeof(double));

    for (int i = 0; i < d; i++)
    {
        int next_len = vals_len - 1;
        for (int j = 0; j < next_len; j++)
        {
            next[j] = vals[j+1] - vals[j];
        }

        vals_len = next_len;

        // Swap buffers for next order differencing
        double *tmp = vals;
        vals = next;
        next = tmp;
    }

    /* Convert to PG return type */
    Datum *darray = palloc(sizeof(Datum) * (n_vals - d));
    for (int i = 0; i < n_vals - d; i++)
    {
        darray[i] = Float8GetDatum(vals[i]);
    }
    ArrayType *pg_differenced = construct_array(darray, n_vals - d, FLOAT8OID,
                                                sizeof(double), FLOAT8PASSBYVAL,
                                                'd');
    PG_RETURN_ARRAYTYPE_P(pg_differenced);
}

// Compute coefficients of integration expansion using the Binomial Theorem for d > -1
// Box, Jenkins, & Reinsel (2008). Time Series Analysis: Forecasting and Control (4th edition, Chapter 10, p.429)
static double* _integration_coefficients(int d) {
    double* c = palloc((d + 1) * sizeof(double));
    long prev = 1;
    for (int k = 1; k <= d; k++) {
        prev = prev * (d - (k - 1)) / k;   // binomial(d, k) from binomial(d, k-1)
        c[k] = (k % 2 ? +prev : -prev);    // apply sign (-1)^(k+1)
    }
    return c;
}

Datum
arima_integrate(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *diff_arr = PG_GETARG_ARRAYTYPE_P(0);
    int32 d = PG_GETARG_INT32(1);
    ArrayType *initial_arr = PG_GETARG_ARRAYTYPE_P(2);

    /* Validate arguments */
    if (ARR_NDIM(diff_arr) != 1 || ARR_NDIM(initial_arr) != 1)
    {
        ereport(ERROR,
                (errmsg("differences and initial_vals must be 1-D arrays")));
    }
    // No differencing required
    if (d <= 0)
    {
        PG_RETURN_ARRAYTYPE_P(diff_arr);
    }

    /* Convert psql array to C array */
    int n_diff, n_initial;
    double* differences = pg_array_to_c_double(diff_arr, &n_diff, false, "arima_integrate");
    double* initial_vals = pg_array_to_c_double(initial_arr, &n_initial, false, "arima_integrate");

    if (n_initial != d)
    {
        ereport(ERROR, (errmsg("arima_integrate: number of initial vals must match differencing order")));
    }

    /* Perform integration */
    double *integrated = palloc0((n_diff + d) * sizeof(double));
    double *coeffs = _integration_coefficients(d);
    
    // Copy initial values to result
    for (int i = 0; i < d; i++)
    {
        integrated[i] = initial_vals[i];
    }

    for (int i = 0; i < n_diff; i++)
    {
        double val = differences[i];
        for (int j = 1; j <= d; j++)
        {
            val += coeffs[j] * integrated[i+d-j];
        }
        integrated[i+d] = val;
    }

    /* Convert to PG return type */
    Datum *darray = palloc(sizeof(Datum) * (n_diff + d));
    for (int i = 0; i < n_diff + d; i++)
    {
        darray[i] = Float8GetDatum(integrated[i]);
    }
    ArrayType *pg_integrated = construct_array(darray, n_diff + d, FLOAT8OID,
                                               sizeof(double), FLOAT8PASSBYVAL,
                                               'd');
    PG_RETURN_ARRAYTYPE_P(pg_integrated);
}

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
    double* vals = pg_array_to_c_double(vals_arr, &n_vals, false, "css_loss");
    double* phi = pg_array_to_c_double(phi_arr, &n_phi, false, "css_loss");
    double* theta = pg_array_to_c_double(theta_arr, &n_theta, false, "css_loss");
    
    /* CSS logic */
    double* resid = palloc(n_vals * sizeof(double));
    PG_RETURN_FLOAT8(css(vals, phi, theta, p, q, n_vals, NULL, resid));
}

/* ARIMA optimisation */
static double _arima_objective_no_grad(unsigned n, const double *x, double *grad, void *data)
{
    css_data_t *d = (css_data_t *)data;
    double *phi = (double *)x;
    double* theta = (double *)(x + d->p);
    double* resid = (double *)(d->resid);

    // Gradient is not required by Nelder-Mead algorithm
    return css(d->vals, phi, theta, d->p, d->q, d->n_vals, NULL, resid);
}

static double _arima_objective_grad(unsigned n, const double *x, double *grad, void *data)
{
    css_data_t *d = (css_data_t *)data;
    double *phi = (double *)x;
    double* theta = (double *)(x + d->p);
    double* resid = (double *)(d->resid);
    
    return css(d->vals, phi, theta, d->p, d->q, d->n_vals, grad, resid);
}

static opt_result_t _arima_nlopt(double* vals, int n_vals, int p, int q,
                                nlopt_algorithm algorithm, const char* algorithm_name,
                                double (*objective)(unsigned, const double*, double*, void*))
{
    int n_params = p + q;
    nlopt_opt opt = nlopt_create(algorithm, n_params);
    double* resid = palloc(n_vals * sizeof(double));

    css_data_t data = {vals, n_vals, p, q, resid};
    nlopt_set_min_objective(opt, objective, &data);

    // Lower and upper bounds (stationarity assumed to be enforced)
    double *lb = palloc(sizeof(double) * n_params);
    double *ub = palloc(sizeof(double) * n_params);
    for (int i = 0; i < n_params; i++)
    {
        lb[i] = ARIMA_OPTIMISER_MIN_BOUND;
        ub[i] = ARIMA_OPTIMISER_MAX_BOUND;
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
        errmsg("NLopt %s failed", algorithm_name));
    }

    nlopt_destroy(opt);
    elog(INFO, "Min CSS: %f", result);

    opt_result_t return_val = {x, resid};
    return return_val;
}

Datum
arima_optimise(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *vals_arr  = PG_GETARG_ARRAYTYPE_P(0);
    int32 p = PG_GETARG_INT32(1);
    int32 q = PG_GETARG_INT32(2);
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
    int n_vals;
    double* vals = pg_array_to_c_double(vals_arr, &n_vals, false, "arima_optimise");

    /* Call optimiser */
    double (*opt_objective)(unsigned, const double *, double *, void *);
    nlopt_algorithm opt_algorithm;
    int n_params = p + q;
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
    else
    {
        elog(ERROR, "Invalid optimiser provided: %s", arima_method);
    }
    opt_result = _arima_nlopt(vals, n_vals, p, q, opt_algorithm, arima_method, opt_objective);

    /* Warn for risky bounds */
    for (int i = 0; i < n_params; i++)
    {
        double val = opt_result.params[i];
        if (val > 1.0 || val < -1.0)
        {
            elog(WARNING, "AR/MA model not invertible: phi/theta = %f", val);
        }
    }

    /* Convert to arima_optimise_result return type */
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
                 errmsg("type \"arima_optimise_result\" does not exist")));
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

// TODO: implement Kalman Filter (same as R)?

/* Forecasting values */
static double* _arima_predict(double* values, int n_vals, double* resid, int p,
                              int q, double* phi, double* theta, unsigned int horizon)
{
    double* yhat = palloc0(horizon * sizeof(double));
    double fcast, ar, ma;

    for (int t = 0; t < horizon; t++)
    {
        fcast = 0.0;
        for (int j = 0; j < p; j++)
        {
            ar = 0.0;
            // If we're forecasting from future values, use yhat
            // (if not, use values provided)
            if (t > j)
            {
                ar = phi[j] * yhat[t-j-1];
            }
            else
            {
                int idx = n_vals - (j+1-t);
                if (idx >= 0) ar = phi[j] * values[idx];
            }
            fcast += ar;
        }
        for (int k = 0; k < q; k++)
        {
            ma = 0.0;
            // If we're forecasting from past values
            // (if not, residual is 0)
            if (t <= k)
            {
                int idx = k-t; // resid[0] = residual for x_{t-0}
                if (idx >= 0) ma = theta[k] * resid[idx];
            }
            fcast += ma;
        }
        yhat[t] = fcast;
    }
    return yhat;
}

Datum
arima_forecast(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *vals_arr  = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *resid_arr  = PG_GETARG_ARRAYTYPE_P(1);
    int32 p = PG_GETARG_INT32(2);
    int32 q = PG_GETARG_INT32(3);
    ArrayType *phi_arr  = PG_GETARG_ARRAYTYPE_P(4);
    ArrayType *theta_arr  = PG_GETARG_ARRAYTYPE_P(5);
    int32 horizon = PG_GETARG_INT32(6);

    /* Validate arguments */
    if (ARR_NDIM(vals_arr) != 1 || ARR_NDIM(resid_arr) != 1 || ARR_NDIM(phi_arr) != 1 || ARR_NDIM(theta_arr) != 1)
    {
        ereport(ERROR,
                (errmsg("vals, residuals, phi and theta must be 1-D arrays")));
    }
    if (p < 1 || q < 1 || horizon < 1)
    {
        ereport(ERROR,
                (errmsg("p, q and horizon must be positive integers")));
    }

    /* Convert psql arrays to C arrays */
    int n_vals, n_resid, n_phi, n_theta;
    double *vals = pg_array_to_c_double(vals_arr, &n_vals, false, "arima_forecast");
    double *resid = pg_array_to_c_double(resid_arr, &n_resid, false, "arima_forecast");
    double *phi = pg_array_to_c_double(phi_arr, &n_phi, false, "arima_forecast");
    double *theta = pg_array_to_c_double(theta_arr, &n_theta, false, "arima_forecast");

    /* Predict */
    double* yhat = _arima_predict(vals, n_vals, resid, p, q, phi, theta, horizon);

    /* Convert to PG return type */
    Datum *darray = palloc(sizeof(Datum) * horizon);
    for (int i = 0; i < horizon; i++)
    {
        darray[i] = Float8GetDatum(yhat[i]);
    }
    ArrayType *pg_yhat = construct_array(darray, horizon, FLOAT8OID,
                                        sizeof(double), FLOAT8PASSBYVAL,
                                        'd');
    PG_RETURN_ARRAYTYPE_P(pg_yhat);
}