#include <stdbool.h>

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

#endif