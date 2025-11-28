#include "postgres.h"
#include "fmgr.h"
#include "utils/array.h"
#include "utils/lsyscache.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "funcapi.h"

PG_FUNCTION_INFO_V1(kpss);
PG_FUNCTION_INFO_V1(aicc);

#define _USE_MATH_DEFINES
#include <math.h>
#include "autoarima.h"
#include "utils.h"
#include "constants.h"

/* KPSS statistical test */
// Newey-West long-run variance using Bartlett kernel for KPSS
// (Kwiatkowski et al., 1992)
static double _long_run_variance(double *resid, int n, int lags)
{
    double s_hat = 0.0;

    // Standard variance of residuals (without considering correlations)
    for (int i = 0; i < n; i++)
        s_hat += resid[i] * resid[i];

    // Autocovariance
    for (int lag = 1; lag <= lags; lag++)
    {
        double sum = 0.0;
        for (int t = lag; t < n; t++)
            sum += resid[t] * resid[t - lag];
        // Bartlett kernel weight
        s_hat += 2.0 * sum * (1.0 - (double)lag / (lags + 1));
    }

    return s_hat / n; // LR variance per observation
}

// KPSS automatic lag selection (Hobijn et al., 1998)
static int _kpss_autolag(double *resid, int n)
{
    // n^(2/9) is a rule of thumb from Hobijn et al.
    int covlags = (int)pow((double)n, 2.0 / 9.0);

    // Standard variance of residuals
    double s0 = 0.0;
    for (int i = 0; i < n; i++)
        s0 += resid[i] * resid[i];
    s0 /= n;

    // Weighted sum of covariances
    double s1 = 0.0;
    for (int lag = 1; lag <= covlags; lag++)
    {
        double prod = 0.0;
        for (int t = lag; t < n; t++)
            prod += resid[t] * resid[t - lag];
        prod /= (n / 2.0); // Normalisation factor
        s0 += prod;
        s1 += lag * prod; // Weighting
    }

    double s_hat = s1 / s0;
    // Intermediate cube root scaling - 1.1447 found by Hobijn
    double gamma_hat = 1.1447 * pow(s_hat * s_hat, 1.0 / 3.0);
    int autolags = (int)(gamma_hat * pow((double)n, 1.0 / 3.0));
    if (autolags >= n) autolags = n - 1;

    return autolags;
}

// KPSS statistic computation
double _kpss(double *x, int n)
{
    double avg = arr_mean(x, n);

    // Residuals
    double* resid = palloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        resid[i] = x[i] - avg;

    // Cumulative sum of residuals
    double* S = palloc(n * sizeof(double));
    S[0] = resid[0];
    for (int i = 1; i < n; i++)
        S[i] = S[i - 1] + resid[i];

    int lags = _kpss_autolag(resid, n);

    double lr_var = _long_run_variance(resid, n, lags);

    // KPSS statistic
    double eta = 0.0;
    for (int i = 0; i < n; i++)
        eta += S[i] * S[i];

    return eta / (n * n * lr_var);
}

Datum
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

/* AICc (Information Criterion for selection) */
double css_loglik(double css, int p, int q, int T)
{
    double sigma2 = css / T;
    // Conditional Gaussian ARMA likelihood
    double loglik = -0.5 * T * (log(2*M_PI) + log(sigma2) + 1.0);
    return loglik;
}

double arima_aic(double loglik, int p, int q, int k)
{
    int num_params = p + q + k + 1;
    return -2.0 * loglik + 2.0 * num_params;
}

/*
    Calculates the corrected Akaike's Information Criterion (AICc)
    https://otexts.com/fpp3/arima-estimation.html

    css    - Conditional Sum of Squares value
    p, q   - ARIMA constants
    k      - 0 if no constant used, 1 otherwise
    n_vals - num of values the model was fitted with
*/
double arima_aicc(double css, int p, int q, int k, int n_vals)
{
    int ncond = max(p, q);
    int T = n_vals - ncond;
    if (T <= 0) return NAN;

    double loglik = css_loglik(css, p, q, T);
    elog(DEBUG1, "Log likelihood: %f", loglik);

    int num_params = p + q + k + 1;

    double aic = arima_aic(loglik, p, q, k);
    elog(DEBUG1, "AIC: %f", aic);

    double correction =
        (2.0 * num_params * (num_params + 1)) /
        (n_vals - num_params - 1.0);

    return aic + correction;
}

Datum
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