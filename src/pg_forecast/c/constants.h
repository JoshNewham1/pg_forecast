#ifndef CONSTANTS_H
#define CONSTANTS_H

#define AR_MAX 10      /* max AR order (phi) */
#define MA_MAX 10      /* max MA order (theta) */
#define D_MAX 2        /* max differencing order (d) */

static const double ARIMA_OPTIMISER_MIN_BOUND = -2.0; // default -2.0
static const double ARIMA_OPTIMISER_MAX_BOUND = 2.0; // default 2.0
static const double ARIMA_OPTIMISER_XTOL_REL = 1e-6; // 0 to disable, default 1e-6

static const double ARIMA_OPTIMISER_BASE_TIME = 15.0; // time (seconds) before timeout for base rows
static const int ARIMA_OPTIMISER_BASE_ROWS = 100000; // default 100k
/* 
    Extra time to give each row above ARIMA_OPTIMISER_BASE_ROWS
    as the loss function takes longer to evaluate.
    Found empirically on 10 million rows 
 */
static const double ARIMA_OPTIMISER_ROW_MULTIPLIER = 0.0000015;

static const int AUTOARIMA_NUM_P_VALUES = 4;
static const double AUTOARIMA_P_VALUES[] = {0.1, 0.05, 0.025, 0.01};
static const double AUTOARIMA_KPSS_CRITICAL_VALS[] = {0.347, 0.463, 0.574, 0.739};

static const double ARIMA_INIT_PARAM = 0.1;
static const double ARIMA_C_SCALE = 10.0;
static const double ARIMA_MIN_C_BOUND = 1.0;

#endif