#ifndef ARIMA_H
#define ARIMA_H

double css(double *vals, double *phi, double *theta, int p, int q, int n_vals, double* grad);

typedef struct {
    double *vals;
    int n_vals;
    int p;
    int q;
} css_data_t;

double *optimise_arima_nelder(double *vals, int n_vals, int p, int q);
double *optimise_arima_lbfgs(double *vals, int n_vals, int p, int q);

#endif