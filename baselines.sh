source ".venv/bin/activate"
cd eval/TFB

# Univariate forecasts
# From scripts/univariate_forecast/univariate-forecast.sh
python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_yearly.json"  --model-name  "darts.LinearRegressionModel" "darts.StatsForecastAutoETS" "darts.AutoARIMA" "darts.XGBModel" "darts.RandomForest" "darts.RNNModel" "time_series_library.PatchTST" --model-hyper-params "{\"d_model\":16,\"d_ff\":32,\"factor\":3}" "{\"d_model\":16,\"d_ff\":32,\"patch_len\":4,\"stride\":2}"  "{\"p_hidden_layers\":2,\"p_hidden_dims\":[256,256],\"d_model\":16,\"d_ff\":32,\"factor\":3}" "{\"d_model\":16,\"d_ff\":32,\"factor\":3}"  "{\"d_model\":16,\"d_ff\":32,\"factor\":3}" "{\"d_model\":16,\"d_ff\":32,\"factor\":3}" "{\"d_model\":16,\"d_ff\":32,\"factor\":3}" "{\"d_model\":16,\"d_ff\":32,\"factor\":3}" "{\"d_model\":16,\"d_ff\":32,\"factor\":3}" "{\"d_model\":16,\"d_ff\":32,\"factor\":3}"  "{\"d_model\":16,\"d_ff\":32,\"factor\":3}" "{\"d_model\":16,\"d_ff\":32,\"factor\":3}" "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}" "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}" "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}" "{\"n_epochs\":5,\"batch_size\":16,\"optimizer_kwargs\":{\"lr\":1e-3}}"  --save-path "yearly"  --gpus 0  --num-workers 1 --timeout 60000

# Multivariate forecasts