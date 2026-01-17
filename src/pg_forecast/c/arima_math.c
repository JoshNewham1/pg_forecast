#include "postgres.h"
#include "utils.h"
#include "arima.h"
#include <math.h>
#include <string.h>

double* arima_compute_difference(double* vals, int n_vals, int d)
{
    /* Perform differencing */
    int vals_len = n_vals;
    double *current_vals = vals;
    double *next = palloc0(vals_len * sizeof(double));
    double *temp_vals = palloc(vals_len * sizeof(double));
    memcpy(temp_vals, vals, vals_len * sizeof(double));
    
    current_vals = temp_vals;

    for (int i = 0; i < d; i++)
    {
        int next_len = vals_len - 1;
        for (int j = 0; j < next_len; j++)
        {
            next[j] = current_vals[j+1] - current_vals[j];
        }

        vals_len = next_len;

        // Swap buffers for next order differencing
        double *tmp = current_vals;
        current_vals = next;
        next = tmp;
    }
    return current_vals;
}

double* arima_compute_integration(double* differences, int n_diff, int d, double* initial_vals)
{
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
    
    pfree(coeffs); // _integration_coefficients pallocs
    return integrated;
}

double* _integration_coefficients(int d) {
    double* c = palloc((d + 1) * sizeof(double));
    long prev = 1;
    for (int k = 1; k <= d; k++) {
        prev = prev * (d - (k - 1)) / k;   // binomial(d, k) from binomial(d, k-1)
        c[k] = (k % 2 ? +prev : -prev);    // apply sign (-1)^(k+1)
    }
    return c;
}

double css(double *vals, double *phi, double *theta, int p, int q, bool include_c, double c, int n_vals, double* grad, double* resid)
{
    int ncond = max(p, q);
    if (resid != NULL) memset(resid, 0, n_vals * sizeof(double));
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

        // AR part
        for (int j = 0; j < p; j++)
        {
            tmp -= phi[j] * vals[t-j-1];
        }

        // MA part
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

static void autocovariance(const double *x, int n, int p, double *gamma)
{
    double mean = 0.0;
    for (int i = 0; i < n; i++)
        mean += x[i];
    mean /= n;

    for (int k = 0; k <= p; k++)
    {
        gamma[k] = 0.0;
        for (int i = 0; i < n - k; i++)
            gamma[k] += (x[i] - mean) * (x[i + k] - mean);
        gamma[k] /= n;
    }
}

double* yule_walker(const double *x, int n, int p)
{
    double *gamma = (double*)palloc((p + 1) * sizeof(double));
    double *phi = (double*)palloc(p * sizeof(double));
    double *rhs = (double*)palloc(p * sizeof(double));

    autocovariance(x, n, p, gamma);

    // Set up the Toeplitz matrix (only need gamma[0..p-1])
    double *R = (double*)palloc(p * p * sizeof(double));
    for (int i = 0; i < p; i++)
    {
        for (int j = 0; j < p; j++)
        {
            R[i * p + j] = gamma[abs(i - j)];
        }
        rhs[i] = gamma[i + 1];
    }

    // Solve R * phi = rhs using naive Gaussian elimination
    for (int i = 0; i < p; i++)
    {
        // pivot
        double pivot = R[i * p + i];
        if (fabs(pivot) < 1e-12) pivot = 1e-12;  // avoid div by zero
        for (int j = i; j < p; j++)
            R[i * p + j] /= pivot;
        rhs[i] /= pivot;

        // eliminate
        for (int k = i + 1; k < p; k++)
        {
            double factor = R[k * p + i];
            for (int j = i; j < p; j++)
                R[k * p + j] -= factor * R[i * p + j];
            rhs[k] -= factor * rhs[i];
        }
    }

    // back-substitution
    for (int i = p - 1; i >= 0; i--)
    {
        phi[i] = rhs[i];
        for (int j = i + 1; j < p; j++)
            phi[i] -= R[i * p + j] * phi[j];
    }

    pfree(gamma);
    pfree(rhs);
    pfree(R);

    return phi;
}

static void compute_residuals(const double *x, int n, const double *phi, int p, double c, double *res)
{
    for (int t = 0; t < n; t++)
    {
        double pred = c;
        for (int i = 1; i <= p; i++) 
        {
            if (t - i >= 0)
            {
                pred += phi[i - 1] * x[t - i];
            }
        }
        res[t] = x[t] - pred;
    }
}

double* ma_initial_guess(const double *x, int n, int p, int q, double c)
{
    if (q == 0) return NULL;

    double *phi = NULL;
    if (p > 0) phi = yule_walker(x, n, p);

    double *res = (double*)palloc(n * sizeof(double));

    compute_residuals(x, n, phi ? phi : NULL, p, c, res);

    // Fit AR(q) on residuals => treat as MA(q) initial guess
    double *theta = yule_walker(res, n, q);

    pfree(res);
    if (phi) pfree(phi);

    return theta;
}

double* _arima_predict(double* values, int n_vals, double* resid, int p,
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