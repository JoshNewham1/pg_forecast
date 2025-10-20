source ".venv/bin/activate"
cd eval/TFB

# Univariate forecasts
# From scripts/univariate_forecast/univariate-forecast.sh
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_yearly.json"  --model-name  "darts.LinearRegressionModel" "darts.StatsForecastAutoETS" "darts.XGBModel" "darts.RandomForest" "darts.AutoARIMA" --save-path "yearly"  --gpus 0  --num-workers 1 --timeout 60000