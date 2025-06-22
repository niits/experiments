#!/bin/bash

set -euxo pipefail;

mkdir -p assets;

if [ ! -f assets/DB_TD500_resnet50.onnx ]; then
    echo "Downloading DB_TD500_resnet50.onnx";
    pip install gdown;
    gdown "https://drive.google.com/uc?export=dowload&id=19YWhArrNccaoSza0CfkXlA8im4-lAGsR" -O assets/DB_TD500_resnet50.onnx;
else
    echo "DB_TD500_resnet50.onnx already exists";
fi



docker run \
    --gpus all \
    -v $(pwd)/assets:/mnt/assets \
    --rm \
    -it \
    nvcr.io/nvidia/tritonserver:24.04-py3  \
        /usr/src/tensorrt/bin/trtexec \
            --onnx=/mnt/assets/DB_TD500_resnet50.onnx \
            --saveEngine=/mnt/assets/model.plan \
            --shapes=input:1x3x960x960 \
            --workspace=3000
