#include "postgres.h"
#include "utils.h"
#include "arima.h"
#include "constants.h"
#include <nlopt.h>
#include <math.h>
#include <string.h>

double _arima_objective_no_grad(unsigned n, const double *x, double *grad, void *data)
{
    css_data_t *d = (css_data_t *)data;
    double *phi = (double *)x;
    double c = d->include_c ? x[d->p + d->q] : 0.0;
    double* theta = (double *)(x + d->p);
    double* resid = (double *)(d->resid);

    // Gradient is not required by Nelder-Mead algorithm
    return css(d->vals, phi, theta, d->p, d->q, d->include_c, c, d->n_vals, NULL, resid);
}

double _arima_objective_grad(unsigned n, const double *x, double *grad, void *data)
{
    css_data_t *d = (css_data_t *)data;
    double *phi = (double *)x;
    double* theta = (double *)(x + d->p);
    double c = d->include_c ? x[d->p + d->q] : 0.0;
    double* resid = (double *)(d->resid);
    
    return css(d->vals, phi, theta, d->p, d->q, d->include_c, c, d->n_vals, grad, resid);
}

opt_result_t _arima_nlopt(double* vals, int n_vals, int p, int q, bool include_c,
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
        double bound = fmax(ARIMA_C_SCALE * abs_mean, ARIMA_MIN_C_BOUND);
        
        lb[p + q] = -bound;
        ub[p + q] = bound;
    }
    nlopt_set_lower_bounds(opt, lb);
    nlopt_set_upper_bounds(opt, ub);

    // Initial guess - all parameters are ARIMA_INIT_PARAM, and intercept is mean
    double *x = palloc(sizeof(double) * n_params);
    for (int i = 0; i < n_params; i++) x[i] = ARIMA_INIT_PARAM;

    double mean = arr_mean(vals, n_vals);

    if (p > 0)
    {
        double *phi_init = yule_walker(vals, n_vals, p);
        for (int i = 0; i < p; i++) x[i] = phi_init[i];
        pfree(phi_init);
    }
    if (q > 0)
    {
        double *theta_init = ma_initial_guess(vals, n_vals, p, q, mean);
        for (int i = 0; i < q; i++) x[p+i] = theta_init[i];
        pfree(theta_init);
    }
    if (include_c)
    {
        x[p + q] = mean;
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