#!/bin/bash

# Assume .venv directory is current, if not specified
# e.g. can use /tmp/.venv for more space
if [ -z "$INDB_VENV" ]; then
  INDB_VENV = ""
fi

# Install TFB Python dependencies
pip install venv
python3 -m venv "$INDB_VENV.venv"
source "$INDB_VENV.venv/bin/activate"
pip install -r requirements.txt

# Ensure eval datasets have been downloaded
TFB_DATASET_DIR="eval/TFB/dataset"
if [ ! -d "$TFB_DATASET_DIR" ]; then
    cat << EOF
--------------------------------------------------------
ERROR: Required TFB dataset directory missing!

The directory was not found at the expected location:
$TFB_DATASET_DIR

Please download the datasets from their repository or use
this direct link:
https://drive.google.com/file/d/1vgpOmAygokoUt235piWKUjfwao6KwLv7/view?usp=drive_link
--------------------------------------------------------
EOF
    exit 1
fi

MONASH_DATASET_DIR="eval/monash/data"
if [ ! -d "$MONASH_DATASET_DIR" ]; then
    cat << EOF
--------------------------------------------------------
ERROR: Monash dataset directory missing!

The directory was not found at the expected location:
$MONASH_DATASET_DIR

Please download the following datasets from their repository or use
this direct link:
https://huggingface.co/datasets/Monash-University/monash_tsf/tree/main/data

* solar_10_minutes_dataset
* wind_farms_minutely_dataset_with_missing_values
--------------------------------------------------------
EOF
    exit 1
fi

# Add TFB adapters (via symlink)
mkdir -p eval/TFB/ts_benchmark/baselines/pg_forecast
find src/TFB/ts_benchmark/baselines/pg_forecast \
  -type f \
  -exec sh -c '
    for f; do
        target="eval/TFB/ts_benchmark/baselines/pg_forecast/${f#src/TFB/ts_benchmark/baselines/pg_forecast/}"
        ln -sf "$(realpath "$f")" "$target"
    done
    ' sh {} +

mkdir -p eval/TFB/ts_benchmark/baselines/python_competitor
find src/TFB/ts_benchmark/baselines/python_competitor \
  -type f \
  -exec sh -c '
    for f; do
        target="eval/TFB/ts_benchmark/baselines/python_competitor/${f#src/TFB/ts_benchmark/baselines/python_competitor/}"
        ln -sf "$(realpath "$f")" "$target"
    done
    ' sh {} +

# Copy over changes to JoinBoost/main (executor.py)
cp src/JoinBoost/src/joinboost/* eval/JoinBoost/src/joinboost/