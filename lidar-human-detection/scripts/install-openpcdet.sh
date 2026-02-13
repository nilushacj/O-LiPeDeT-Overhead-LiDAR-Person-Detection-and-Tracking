#!/bin/sh

set -e

source scripts/source.sh
conda env update -f environment.yml
pushd vendor/OpenPCDet
pip install -e . --no-build-isolation
popd

trap 'on_error' ERR

on_error() {
    echo "An error occurred during the installation."
    popd || true
    conda deactivate
}
