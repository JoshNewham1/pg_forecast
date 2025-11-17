#ifndef ARIMA_H
#define ARIMA_H

double css(double *vals, double *phi, double *theta, int p, int q, int n_vals);

typedef struct {
    double *vals;
    int n_vals;
    int p;
    int q;
} css_data_t;

double arima_nelder_objective(unsigned n, const double *x, double *grad, void *data);
double *optimise_arima_nelder(double *vals, int n_vals, int p, int q);

#endif