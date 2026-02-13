#!/bin/sh

source ~/.bashrc

export MY_ENV_NAME="openpcdet"
# export ENV_NAME="lidar-human-detection-models"

if conda env list | grep -q "$MY_ENV_NAME"; then
    conda activate "$MY_ENV_NAME"
fi

# PYTHON_VERSION_=$(python --version | cut -d ' ' -f 2 | cut -d '.' -f 1-2)
# export PYTHONPATH="${CONDA_PREFIX}/lib/python${PYTHON_VERSION_}/site-packages:$PYTHONPATH"
