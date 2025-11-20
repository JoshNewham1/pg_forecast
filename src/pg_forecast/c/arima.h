#ifndef ARIMA_H
#define ARIMA_H

double css(double *vals, double *phi, double *theta, int p, int q, int n_vals, double* grad, double* resid);

typedef struct {
    double *vals;
    int n_vals;
    int p;
    int q;
    double *resid;
} css_data_t;

typedef struct {
    double *params;
    double *resid;
} opt_result_t;

opt_result_t optimise_arima_nelder(double *vals, int n_vals, int p, int q);
opt_result_t optimise_arima_lbfgs(double *vals, int n_vals, int p, int q);

#endif