#!/bin/bash

# Assume .venv directory is current, if not specified
# e.g. can use /tmp/.venv for more space
if [ -z "$INDB_VENV" ]; then
  INDB_VENV = ""
fi

# Install TFB Python dependencies
pip install virtualenv
python3 -m virtualenv "$INDB_VENV.venv"
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