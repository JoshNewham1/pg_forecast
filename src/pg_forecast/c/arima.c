#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "catalog/pg_type.h"

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

double css(double *vals, double *phi, double *theta, int p, int q, int n_vals)
{
    int ncond = max(p, q);
    double* resid = palloc(n_vals * sizeof(double));
    memset(resid, 0.0, n_vals * sizeof(double));
    double ssq = 0.0;

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
        if (!isnan(tmp))
        {
            ssq += tmp * tmp;
        }
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
    PG_RETURN_FLOAT8(css(vals, phi, theta, p, q, n_vals));
}

Datum
optimise_arima(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *vals_arr  = PG_GETARG_ARRAYTYPE_P(0);
    int32 p = PG_GETARG_INT32(1);
    int32 q = PG_GETARG_INT32(2);
    int n_vals = ArrayGetNItems(ARR_NDIM(vals_arr), ARR_DIMS(vals_arr));

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
    double* opt_result = optimise_arima_nelder(vals, n_vals, p, q);

    /* Warn for risky bounds */
    int result_len = p + q;
    for (int i = 0; i < result_len; i++)
    {
        double val = opt_result[i];
        if (val > 1.0 || val < -1.0)
        {
            elog(WARNING, "phi/theta coefficient outside of [-1, 1]: %f\n", val);
        }
    }

    /* Convert to PG return type */
    Datum *darray = palloc(sizeof(Datum) * result_len);
    for (int i = 0; i < result_len; i++)
    {
        darray[i] = Float8GetDatum(opt_result[i]);
    }
    ArrayType *pg_opt_result = construct_array(darray, result_len, FLOAT8OID,
                                               sizeof(double), FLOAT8PASSBYVAL,
                                               'd');
    PG_RETURN_ARRAYTYPE_P(pg_opt_result);
}

double arima_nelder_objective(unsigned n, const double *x, double *grad, void *data)
{
    css_data_t *d = (css_data_t *)data;
    double *phi = (double *)x;
    double* theta = (double *)(x + d->p);

    // Gradient is not required by Nelder-Mead algorithm
    return css(d->vals, phi, theta, d->p, d->q, d->n_vals);
}

double *optimise_arima_nelder(double *vals, int n_vals, int p, int q)
{
    int n_params = p + q;
    nlopt_opt opt = nlopt_create(NLOPT_LN_NELDERMEAD, n_params);

    css_data_t data = {vals, n_vals, p, q};
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

    return x;
}