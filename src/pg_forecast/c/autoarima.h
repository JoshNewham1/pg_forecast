#ifndef AUTOARIMA_H
#define AUTOARIMA_H

double _kpss(double *x, int n);

typedef struct {
    double kpss_val;
    double crit_val;
    bool test_passed;
} kpss_result_t;

double css_loglik(double css, int p, int q, int n_vals);
double arima_aic(double loglik, int p, int q, int k);
double arima_aicc(double loglik, int p, int q, int k, int T);

#endif