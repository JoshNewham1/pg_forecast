#include <stdbool.h>
#include <nlopt.h>

#ifndef ARIMA_H
#define ARIMA_H

typedef struct {
    double *vals;
    int n_vals;
    int p;
    int q;
    bool include_c;
    double *resid;
} css_data_t;

typedef struct {
    double *params;
    double *resid;
    double css;
} opt_result_t;

typedef struct {
    int t; // Current time index
    int p;
    int q;
    double css;
    double *y_lags;
    double *e_lags;
} arima_inc_state_t;

double css(double *vals, double *phi, double *theta, int p, int q, bool include_c, double c, int n_vals, double* grad, double* resid);

double* arima_compute_difference(double* vals, int n_vals, int d);
double* _integration_coefficients(int d);
double* arima_compute_integration(double* differences, int n_diff, int d, double* initial_vals);

double* _arima_predict(double* values, int n_vals, double* resid, int p,
                              int q, double c, double* phi, double* theta, unsigned int horizon);

opt_result_t _arima_nlopt(double* vals, int n_vals, int p, int q, bool include_c,
                                nlopt_algorithm algorithm, const char* algorithm_name,
                                double (*objective)(unsigned, const double*, double*, void*));
double _arima_objective_no_grad(unsigned n, const double *x, double *grad, void *data);
double _arima_objective_grad(unsigned n, const double *x, double *grad, void *data);

double* yule_walker(const double *x, int n, int p);
double* ma_initial_guess(const double *x, int n, int p, int q, double c);

#endif