# Assume .venv directory is current, if not specified
# e.g. can use /tmp/.venv for more space
if [ -z "$INDB_VENV" ]; then
  INDB_VENV = ""
fi

source "$INDB_VENV.venv/bin/activate"

pytest tests/python/unit_tests.py
pytest tests/python/performance_tests.py --log-cli-level=INFO

# Univariate forecasts
# From scripts/univariate_forecast/univariate-forecast.sh
FREQUENCIES=("yearly" "quarterly" "monthly" "weekly" "daily" "hourly" "other")
for freq in "${FREQUENCIES[@]}"; do
    echo "Processing: $freq benchmark..."

    # Dynamically construct the file paths
    CONFIG_PATH="fixed_forecast_config_${freq}.json"
    SAVE_PATH="$freq"

    # Execute the Python command without the --model-name argument.
    # $PROPERTIES passes any extra arguments provided by the user.
    python ./eval/TFB/scripts/run_benchmark.py --config-path "$CONFIG_PATH" --save-path "$SAVE_PATH" --model-name  "pg_forecast.ARIMA"
done