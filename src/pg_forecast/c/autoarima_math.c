#include "postgres.h"
#include "autoarima.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include "utils.h"
#include "postgres.h"

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