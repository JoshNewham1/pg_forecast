GPU_PROPERTIES='--eval-backend ray --max-tasks-per-child 9999 --gpus 0 --timeout 60000'
CPU_PROPERTIES='--num-workers 1 --timeout 60000'
PROPERTIES=''
USE_GPU=0

# Use GPU properties if -g or --gpu flags specified
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -g|--gpu)
      USE_GPU="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ $USE_GPU -ne 0 ]; then
  PROPERTIES=$GPU_PROPERTIES
else
  PROPERTIES=$CPU_PROPERTIES
fi

# Main script
source "$INDB_VENV.venv/bin/activate"
cd eval/TFB

# Univariate forecasts
# From scripts/univariate_forecast/univariate-forecast.sh
FREQUENCIES=("yearly" "quarterly", "monthly" "weekly" "daily" "hourly" "other")
for freq in "${FREQUENCIES[@]}"; do
    echo "Processing: $freq benchmark..."

    # Dynamically construct the file paths
    CONFIG_PATH="fixed_forecast_config_${freq}.json"
    SAVE_PATH="$freq"

    # Execute the Python command without the --model-name argument.
    # $PROPERTIES passes any extra arguments provided by the user.
    python ./scripts/run_benchmark.py \
        --config-path "$CONFIG_PATH" \
        --save-path "$SAVE_PATH" \
	--model-name  "darts.LinearRegressionModel" "darts.StatsForecastAutoETS" "darts.XGBModel" "darts.RandomForest" "darts.AutoARIMA" \
        $PROPERTIES
done
