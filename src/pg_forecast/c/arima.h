#include <stdbool.h>

#ifndef ARIMA_H
#define ARIMA_H

double css(double *vals, double *phi, double *theta, int p, int q, bool include_c, double c, int n_vals, double* grad, double* resid);

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

#endif