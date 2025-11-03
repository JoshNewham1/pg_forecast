# Common utils

from ts_benchmark.baselines.pg_forecast import ARIMA

# List of model factories to test
MODEL_FACTORIES = [
    ARIMA["model_factory"],
]
