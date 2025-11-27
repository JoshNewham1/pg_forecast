#ifndef AUTOARIMA_H
#define AUTOARIMA_H

double _kpss(double *x, int n);

typedef struct {
    double kpss_val;
    double crit_val;
    bool test_passed;
} kpss_result_t;

#endif