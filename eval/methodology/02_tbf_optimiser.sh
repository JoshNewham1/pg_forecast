# Change `arima.sql` to a different default optimiser (in all locations)
# Change `constants.c` to modify the timeout
# and then compile pg_forecast & run this to get the benchmark results
if [ -z "$INDB_VENV" ]; then
  INDB_VENV = "../../"
fi

source "$INDB_VENV.venv/bin/activate"
python ./eval/TFB/scripts/run_benchmark.py --config-path fixed_forecast_config_daily.json \
                                           --save-path pg_forecast \
                                           --model-name pg_forecast.ARIMA \
                                           --timeout 60000