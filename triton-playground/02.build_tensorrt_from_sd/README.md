# Build TensorRT from Hugging Face Stable Diffusion

This directory contains the necessary files and instructions to build TensorRT from Hugging Face Stable Diffusion

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed the latest version of `Docker` and `Docker Compose`.
- Nvidia GPU with CUDA 12.2 and cuDNN 8.2.2
- Nvidia Container Toolkit installed

## Steps to build TensorRT from Hugging Face Stable Diffusion

### Step 1: Create a Conda Environment

In order to build TensorRT from Hugging Face Stable Diffusion, you need to create a Conda environment with the necessary dependencies.

```bash
conda env create -f environment.yml
```

Due to some issues with current implementations of Diffusion models, you need to install this customized version of `diffusers` package in `custom-diffusers` directory by running the following command:

```bash
pip install custom-diffusers -e
```

### Step 2: Export the Hugging Face Stable Diffusion model to ONNX

Please refer to main notebook for instructions on how to export the model to ONNX at [main.ipynb](./main.ipynb)

Please note that the unet model must be modified due to the Triton issue with 1-D tensors. The model must be modified to accept 2-D tensors. For more information, please refer to the [Triton issue](https://github.com/triton-inference-server/server/issues/5319)

After exporting the model to ONNX, you can find the model in the `exported-models` directory.

```bash
onnx
├── stable-diffusion-v1-5
│   ├── feature_extractor
│   │   └── preprocessor_config.json
│   ├── model_index.json
│   ├── scheduler
│   │   └── scheduler_config.json
│   ├── text_encoder
│   │   ├── config.json
│   │   └── model.onnx
│   ├── tokenizer
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── unet
│   │   ├── config.json
│   │   ├── model.onnx
│   │   └── model.onnx_data
│   ├── vae_decoder
│   │   ├── config.json
│   │   └── model.onnx
│   └── vae_encoder
│       ├── config.json
│       └── model.onnx
├── unet.onnx
└── unet.onnx_data
```

### Step 3: Build TensorRT from exported ONNX models

After exporting the model to ONNX, you can build TensorRT from the exported ONNX models by running the following command:

```bash
docker compose up -d
```

Attach shell to the workspace container and run the following command to build TensorRT from the exported ONNX models:

```bash
cd /mnt/workspace;

bash convert.sh
```

After building TensorRT from the exported ONNX models, you can find the TensorRT models in the `tensorrt_engines` directory. Note that triton server will look for default name `model.plan` in each model directory, so you need to rename the model to `model.plan` before deploying the server. Expected directory structure is as follows:

```bash
models
├── text_encoder
│   ├── 1
│   │   └── model.plan
│   └── config.pbtxt
├── unet
│   ├── 1
│   │   └── model.plan
│   └── config.pbtxt
└── vae_decoder
    ├── 1
    │   └── model.plan
    └── config.pbtxt
```

### Step 4: Deploy the server

After building TensorRT from the exported ONNX models, you can deploy the server using Docker Compose:

```bash
docker compose up
```

### Implement with triton client

TBD
