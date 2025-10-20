#!/bin/bash

# Install TFB Python dependencies
pip install virtualenv
python3 -m virtualenv .venv
source ".venv/bin/activate"
pip install -r eval/TFB/requirements.txt

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