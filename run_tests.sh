# Assume .venv directory is current, if not specified
# e.g. can use /tmp/.venv for more space
if [ -z "$INDB_VENV" ]; then
  INDB_VENV = ""
fi

source "$INDB_VENV.venv/bin/activate"

pytest tests/python/integration_tests.py
pytest tests/python/unit_tests.py
pytest tests/python/performance_tests.py --log-cli-level=INFO
(python eval/competitors/python_autoarima/main.py & python eval/TFB/scripts/run_benchmark.py --config-path fixed_forecast_config_daily.json --save-path python_co
mpetitor --model-name python_competitor.PYTHON_NAIVE & wait)