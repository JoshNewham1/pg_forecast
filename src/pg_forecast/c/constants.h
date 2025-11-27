#ifndef CONSTANTS_H
#define CONSTANTS_H

static const double ARIMA_OPTIMISER_MIN_BOUND = -2.0;
static const double ARIMA_OPTIMISER_MAX_BOUND = 2.0;

static const int AUTOARIMA_NUM_P_VALUES = 4;
static const double AUTOARIMA_P_VALUES[] = {0.1, 0.05, 0.025, 0.01};
static const double AUTOARIMA_KPSS_CRITICAL_VALS[] = {0.347, 0.463, 0.574, 0.739};

#endif