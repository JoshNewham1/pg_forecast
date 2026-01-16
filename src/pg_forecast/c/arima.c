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

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(arima_difference);
PG_FUNCTION_INFO_V1(arima_integrate);
PG_FUNCTION_INFO_V1(css_loss);
PG_FUNCTION_INFO_V1(css_incremental_transition);
PG_FUNCTION_INFO_V1(_css_incremental_transition);
PG_FUNCTION_INFO_V1(_css_incremental_final);
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
    ArrayType *pg_differenced = build_float8_array(vals, n_vals - d);
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
    ArrayType *pg_integrated = build_float8_array(integrated, n_diff + d);
    PG_RETURN_ARRAYTYPE_P(pg_integrated);
}

/* ARIMA loss function */
/* 
  Calculates conditional sum of squares (CSS) and optionally the gradient
  If the gradient isn't required, leave `grad` = NULL
  If the gradient is required, initialise `grad` to hold p+q values
*/
double css(double *vals, double *phi, double *theta, int p, int q, bool include_c, double c, int n_vals, double* grad, double* resid)
{
    int ncond = max(p, q);
    memset(resid, 0, n_vals * sizeof(double));
    double ssq = 0.0;
    double **dphi = NULL, **dtheta = NULL;
    double *dc = NULL;

    if (grad != NULL)
    {
        memset(grad, 0, (p + q + include_c) * sizeof(double));
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
        if (include_c)
        {
            dc = palloc0(sizeof(double) * n_vals);
        }
    }

    for (int t = ncond; t < n_vals; t++)
    {
        double tmp = vals[t] - c;

        // AR part: subtract phi_j * y_{t-j-1}
        for (int j = 0; j < p; j++)
        {
            tmp -= phi[j] * vals[t-j-1];
        }

        // MA part: subtract theta_j * resid_{t-j-1}
        int max_ma = min(t - ncond, q);
        for (int j = 0; j < max_ma; j++)
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
            
            // e_t = y_t - c - sum phi_i*y_{t-i} - sum theta_j*e_{t-j}
            // de_t/dc = -1 - sum theta_j * (de_{t-j}/dc)
            // By the chain rule as the past residuals e_{t-j} also depend on c
            if (include_c && isfinite(tmp))
            {
                double d = -1.0;
                // Add MA contribution
                for (int j = 0; j < max_ma; j++)
                {
                    d -= theta[j] * dc[t-j-1];
                }
                dc[t] = d;
                if (isfinite(d))
                {
                    grad[p + q] += 2.0 * tmp * dc[t];
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
        if (include_c) pfree(dc);
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
    float8 c = PG_GETARG_FLOAT8(5);

    /* Validate arguments */
    if (ARR_NDIM(vals_arr) > 1 || ARR_NDIM(phi_arr) > 1 || ARR_NDIM(theta_arr) > 1)
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
    PG_RETURN_FLOAT8(css(vals, phi, theta, p, q, false, c, n_vals, NULL, resid));
}

static arima_inc_state_t* css_incremental(arima_inc_state_t *state, double* phi, double* theta, double y_t, double c)
{
    double e_t = y_t - c;
    int p = state->p;
    int q = state->q;
    int ncond = max(p, q);

    // AR part
    for (int j = 0; j < p; j++)
    {
        e_t -= phi[j] * state->y_lags[j];
    }

    // MA part
    int max_ma = min(state->t - ncond, q);
    if (max_ma > 0)
    {
        for (int j = 0; j < max_ma; j++)
        {
            e_t -= theta[j] * state->e_lags[j];
        }
    }

    // Skip first ncond terms
    if (state->t >= ncond)
    {
        state->css += e_t * e_t;
    }
    state->t++;

    /* 
        Shift lag arrays forwards one (overwriting last element)
        and add y_t and e_t to front
    */
    if (p > 0)
    {
        memmove(&state->y_lags[1], // Dest
                &state->y_lags[0], // Source
                sizeof(double) * (p - 1)); // Copy first p-1 entries
        state->y_lags[0] = y_t;
    }

    if (q > 0)
    {
        for (int j = 0; j < q - 1; j++)
        {
            state->e_lags[j + 1] = state->e_lags[j];
        }

        // Store current residual at the front
        state->e_lags[0] = e_t;
    }

    return state;
}

static Datum _css_incremental_build_record(arima_inc_state_t *state, TupleDesc tupdesc)
{
    Datum values[6];
    bool nulls[6] = {false};

    values[0] = Int32GetDatum(state->t);
    values[1] = Int32GetDatum(state->p);
    values[2] = Int32GetDatum(state->q);
    values[3] = PointerGetDatum(build_float8_array(state->y_lags, state->p));
    values[4] = PointerGetDatum(build_float8_array(state->e_lags, state->q));
    values[5] = Float8GetDatum(state->css);

    HeapTuple tuple = heap_form_tuple(tupdesc, values, nulls);
    return HeapTupleGetDatum(tuple);
}

static arima_inc_state_t* css_incremental_init(int p, int q)
{
    arima_inc_state_t *state = palloc0(sizeof(arima_inc_state_t));

    state->t = 0;
    state->p = p;
    state->q = q;
    state->css = 0.0;
    state->y_lags = palloc0(sizeof(double) * (p == 0 ? 1 : p));
    state->e_lags = palloc0(sizeof(double) * (q == 0 ? 1 : q));

    return state;
}

Datum
css_incremental_transition(PG_FUNCTION_ARGS)
{
    HeapTupleHeader tup_header = PG_GETARG_HEAPTUPLEHEADER(0); // css_incremental_state arg
    Oid tup_type = HeapTupleHeaderGetTypeId(tup_header);
    int tup_typmod = HeapTupleHeaderGetTypMod(tup_header);
    TupleDesc tupdesc = lookup_rowtype_tupdesc(tup_type, tup_typmod);
    HeapTupleData tup_data;
    tup_data.t_len = HeapTupleHeaderGetDatumLength(tup_header);
    tup_data.t_data = tup_header;

    /*
        Get all values from the css_incremental_state record
        and put into a struct
    */
    bool is_null;
    int t = DatumGetInt32(heap_getattr(&tup_data, 1, tupdesc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: t cannot be NULL")));
    int p = DatumGetInt32(heap_getattr(&tup_data, 2, tupdesc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: p cannot be NULL")));
    int q = DatumGetInt32(heap_getattr(&tup_data, 3, tupdesc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: q cannot be NULL")));

    ArrayType *y_lags_arr = DatumGetArrayTypeP(heap_getattr(&tup_data, 4, tupdesc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: y_lags cannot be NULL")));
    ArrayType *e_lags_arr = DatumGetArrayTypeP(heap_getattr(&tup_data, 5, tupdesc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: e_lags cannot be NULL")));
    double *y_lags = (double *) ARR_DATA_PTR(y_lags_arr);
    double *e_lags = (double *) ARR_DATA_PTR(e_lags_arr);

    double css = DatumGetFloat8(heap_getattr(&tup_data, 6, tupdesc, &is_null));
    if (is_null) ereport(ERROR, (errmsg("css_incremental: css cannot be NULL")));
    
    arima_inc_state_t *state = palloc0(sizeof(arima_inc_state_t));
    state->t = t;
    state->p = p;
    state->q = q;
    state->css = css;
    state->y_lags = y_lags;
    state->e_lags = e_lags;

    // Fetch other args (y, c, phi, theta)
    ArrayType *phi_arr, *theta_arr;
    float8 y_t = PG_GETARG_FLOAT8(1);
    phi_arr = PG_GETARG_ARRAYTYPE_P(2);
    theta_arr = PG_GETARG_ARRAYTYPE_P(3);
    float8 c = PG_GETARG_FLOAT8(4);
    double *phi = (double *) ARR_DATA_PTR(phi_arr);
    double *theta = (double *) ARR_DATA_PTR(theta_arr);

    int n_phi   = ArrayGetNItems(ARR_NDIM(phi_arr), ARR_DIMS(phi_arr));
    int n_theta = ArrayGetNItems(ARR_NDIM(theta_arr), ARR_DIMS(theta_arr));
    if (n_phi != p)
    {
        ereport(ERROR,
                (errmsg("phi arr must have p elements")));
    }
    if (n_theta != q)
    {
        ereport(ERROR,
                (errmsg("theta arr must have q elements")));
    }

    state = css_incremental(state, phi, theta, y_t, c);

    ReleaseTupleDesc(tupdesc);
    PG_RETURN_DATUM(_css_incremental_build_record(state, tupdesc));
}

Datum
_css_incremental_transition(PG_FUNCTION_ARGS)
{
    ArrayType *phi_arr, *theta_arr;
    float8 y_t = PG_GETARG_FLOAT8(1);
    phi_arr = PG_GETARG_ARRAYTYPE_P(2);
    theta_arr = PG_GETARG_ARRAYTYPE_P(3);
    float8 c = PG_GETARG_FLOAT8(4);

    arima_inc_state_t *state;
    if (PG_ARGISNULL(0))
    {
        int p = ArrayGetNItems(ARR_NDIM(phi_arr), ARR_DIMS(phi_arr));
        int q = ArrayGetNItems(ARR_NDIM(theta_arr), ARR_DIMS(theta_arr));

        state = css_incremental_init(p, q);
    }
    else
    {
        state = (arima_inc_state_t *) PG_GETARG_POINTER(0);
    }

    double *phi = (double *) ARR_DATA_PTR(phi_arr);
    double *theta = (double *) ARR_DATA_PTR(theta_arr);

    int n_phi   = ArrayGetNItems(ARR_NDIM(phi_arr), ARR_DIMS(phi_arr));
    int n_theta = ArrayGetNItems(ARR_NDIM(theta_arr), ARR_DIMS(theta_arr));
    if (n_phi != state->p)
    {
        ereport(ERROR,
                (errmsg("phi arr must have p elements")));
    }
    if (n_theta != state->q)
    {
        ereport(ERROR,
                (errmsg("theta arr must have q elements")));
    }

    state = css_incremental(state, phi, theta, y_t, c);
    PG_RETURN_POINTER(state);
}

Datum
_css_incremental_final(PG_FUNCTION_ARGS)
{
    arima_inc_state_t *state = (arima_inc_state_t *) PG_GETARG_POINTER(0);
    TupleDesc tupdesc;

    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        elog(ERROR, "return type must be composite");

    PG_RETURN_DATUM(_css_incremental_build_record(state, tupdesc));
}

/* ARIMA optimisation */
static double _arima_objective_no_grad(unsigned n, const double *x, double *grad, void *data)
{
    css_data_t *d = (css_data_t *)data;
    double *phi = (double *)x;
    double c = d->include_c ? x[d->p + d->q] : 0.0;
    double* theta = (double *)(x + d->p);
    double* resid = (double *)(d->resid);

    // Gradient is not required by Nelder-Mead algorithm
    return css(d->vals, phi, theta, d->p, d->q, d->include_c, c, d->n_vals, NULL, resid);
}

static double _arima_objective_grad(unsigned n, const double *x, double *grad, void *data)
{
    css_data_t *d = (css_data_t *)data;
    double *phi = (double *)x;
    double* theta = (double *)(x + d->p);
    double c = d->include_c ? x[d->p + d->q] : 0.0;
    double* resid = (double *)(d->resid);
    
    return css(d->vals, phi, theta, d->p, d->q, d->include_c, c, d->n_vals, grad, resid);
}

static opt_result_t _arima_nlopt(double* vals, int n_vals, int p, int q, bool include_c,
                                double* phi_guess, double* theta_guess,
                                nlopt_algorithm algorithm, const char* algorithm_name,
                                double (*objective)(unsigned, const double*, double*, void*))
{
    int n_params = p + q + (include_c ? 1 : 0);
    nlopt_opt opt = nlopt_create(algorithm, n_params);
    double* resid = palloc(n_vals * sizeof(double));

    css_data_t data = {vals, n_vals, p, q, include_c, resid};
    nlopt_set_min_objective(opt, objective, &data);

    // Lower and upper bounds (stationarity assumed to be enforced)
    double *lb = palloc(sizeof(double) * n_params);
    double *ub = palloc(sizeof(double) * n_params);
    for (int i = 0; i < n_params; i++)
    {
        lb[i] = ARIMA_OPTIMISER_MIN_BOUND;
        ub[i] = ARIMA_OPTIMISER_MAX_BOUND;
    }

    if (include_c)
    {
        // Scale bounds to be symmetric around mean
        double mean = arr_mean(vals, n_vals);
        double abs_mean = fabs(mean);
        double bound = fmax(10.0 * abs_mean, 1.0);  // At least +/-1.0
        
        lb[p + q] = -bound;
        ub[p + q] = bound;
    }
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);

    // Initial guess - phi/theta are 0.1 or set to provided values, and intercept is mean
    double *x = palloc(sizeof(double) * n_params);
    if (phi_guess != NULL && theta_guess != NULL)
    {
        for (int i = 0; i < p; i++) x[i] = phi_guess[i];
        for (int j = 0; j < q; j++) x[p+j] = theta_guess[j];
    }
    else
    {
        for (int i = 0; i < n_params; i++) x[i] = 0.1;
    }
    if (include_c)
    {
        x[p + q] = arr_mean(vals, n_vals);
    }

    double result;
    if (nlopt_optimize(opt, x, &result) < 0)
    {
        ereport(ERROR,
        errmsg("NLopt %s failed", algorithm_name));
    }

    nlopt_destroy(opt);

    opt_result_t return_val = {x, resid, result};
    return return_val;
}

Datum
arima_optimise(PG_FUNCTION_ARGS)
{
    /* Get arguments */
    ArrayType *vals_arr  = PG_GETARG_ARRAYTYPE_P(0);
    int32 p = PG_GETARG_INT32(1);
    int32 q = PG_GETARG_INT32(2);
    bool include_c = PG_GETARG_BOOL(3);
    char* arima_method = text_to_cstring(PG_GETARG_TEXT_P(4));
    ArrayType *phi_guess_arr = PG_GETARG_ARRAYTYPE_P(5);
    ArrayType *theta_guess_arr = PG_GETARG_ARRAYTYPE_P(6);

    /* Validate arguments */
    if (ARR_NDIM(vals_arr) != 1)
    {
        ereport(ERROR,
                (errmsg("vals must be a 1-D array")));
    }
    if (p < 0 || q < 0)
    {
        ereport(ERROR,
                (errmsg("p and q must be positive integers")));
    }

    /* Convert psql array to C array */
    int n_vals, n_phi_guess, n_theta_guess;
    double* phi_guess = NULL;
    double* theta_guess = NULL;
    double* vals = pg_array_to_c_double(vals_arr, &n_vals, false, "arima_optimise");
    if (!PG_ARGISNULL(5) && !PG_ARGISNULL(6) && ARR_NDIM(phi_guess_arr) == 1 && ARR_NDIM(theta_guess_arr) == 1)
    {
        phi_guess = pg_array_to_c_double(phi_guess_arr, &n_phi_guess, false, "arima_optimise");
        theta_guess = pg_array_to_c_double(theta_guess_arr, &n_theta_guess, false, "arima_optimise");
        if (n_phi_guess != p) ereport(ERROR, (errmsg("phi_guess must be p elements")));
        if (n_theta_guess != q) ereport(ERROR, (errmsg("theta_guess must be q elements")));
    }

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
    else
    {
        elog(ERROR, "Invalid optimiser provided: %s", arima_method);
    }
    opt_result = _arima_nlopt(vals, n_vals, p, q, include_c, phi_guess, theta_guess, opt_algorithm, arima_method, opt_objective);

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
    Datum *phi_array = palloc(sizeof(Datum) * p);
    for (int i = 0; i < p; i++)
    {
        phi_array[i] = Float8GetDatum(opt_result.params[i]);
    }
    ArrayType *phi_converted = construct_array(phi_array, p, FLOAT8OID, sizeof(double), FLOAT8PASSBYVAL, 'd');
    values[0] = PointerGetDatum(phi_converted);

    // Theta array
    Datum *theta_array = palloc(sizeof(Datum) * q);
    for (int i = 0; i < q; i++)
    {
        theta_array[i] = Float8GetDatum(opt_result.params[p+i]);
    }
    ArrayType *theta_converted = construct_array(theta_array, q, FLOAT8OID, sizeof(double), FLOAT8PASSBYVAL, 'd');
    values[1] = PointerGetDatum(theta_converted);

    // Constant term
    values[2] = Float8GetDatum(c);

    // Residuals array
    Datum *resid_array = palloc(sizeof(Datum) * q);
    for (int i = 0; i < q; i++)
    {
        resid_array[i] = Float8GetDatum(opt_result.resid[n_vals-i-1]);
    }
    ArrayType *resid_converted = construct_array(resid_array, q, FLOAT8OID, sizeof(double), FLOAT8PASSBYVAL, 'd');
    values[3] = PointerGetDatum(resid_converted);

    // CSS
    values[4] = Float8GetDatum(opt_result.css);

    tuple = heap_form_tuple(tupdesc, values, nulls);
    result = HeapTupleGetDatum(tuple);
    ReleaseTupleDesc(tupdesc);
    PG_RETURN_DATUM(result);
}

// TODO: implement Kalman Filter (same as R)?

/* Forecasting values */
static double* _arima_predict(double* values, int n_vals, double* resid, int p,
                              int q, double c, double* phi, double* theta, unsigned int horizon)
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
        yhat[t] = fcast + c; // Add intercept if provided
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
    float8 c = PG_GETARG_FLOAT8(4);
    ArrayType *phi_arr  = PG_GETARG_ARRAYTYPE_P(5);
    ArrayType *theta_arr  = PG_GETARG_ARRAYTYPE_P(6);
    int32 horizon = PG_GETARG_INT32(7);

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
    double* yhat = _arima_predict(vals, n_vals, resid, p, q, c, phi, theta, horizon);

    /* Convert to PG return type */
    ArrayType *pg_yhat = build_float8_array(yhat, horizon);
    PG_RETURN_ARRAYTYPE_P(pg_yhat);
}