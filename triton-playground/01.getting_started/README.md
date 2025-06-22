# Triton Server with Conda Environment

This repository contains the necessary configuration and guidelines to deploy a Triton Inference Server with a packed Conda environment using Docker.

## Create a Conda Environment

In order to deploy the server, you need to create a Conda environment with the necessary dependencies. Before installing the packages in your conda environment, make sure that you have exported `PYTHONNOUSERSITE` environment variable:

```bash
export PYTHONNOUSERSITE=True
```

If this variable is not exported and similar packages are installed outside your conda environment, your tar file may not contain all the dependencies required for an isolated Python environment. You can create a new environment using the `environment.yml` file provided in this repository:

```bash
conda env create -f environment.yml
```

## Packing the Conda Environment

Before deploying the server, you need to pack your Conda environment. This is necessary because the server will be running in a Docker container, and it needs to have access to the same Python environment that you used to develop your models.

Here are the steps to pack your Conda environment:

1. Activate your Conda environment:

    ```bash
    conda activate triton-huggingface
    ```

2. Install the `libstdcxx-ng` package from the `conda-forge` channel. This package is a GCC compatibility library that might be necessary for running your models:

    ```bash
    conda install -c conda-forge libstdcxx-ng=12 -y
    ```

3. Pack your Conda environment using the `conda-pack` command. This will create a `.tar.gz` file that contains your environment:

    ```bash
    conda-pack -n triton -o models/python_vit/python3.10.12.tar.gz
    ```

    This command packs the `triton` environment into a `.tar.gz` file located at `models/python_vit/python3.11.tar.gz`.

4. To accelerate the loading time of model_a, you can follow the steps below to unpack the conda environment in the model folder:

    ```bash
    cd models/python_vit;
    mkdir python3.10.12;
    tar -xvf python3.10.12.tar.gz -C  python3.10.12;
    ```

## Deploying the Server

After packing your Conda environment, you can deploy the server using Docker Compose:

```bash
docker compose up
```

## Checking Server Health

Once the server is up and running, you can check its health by sending a request to the /v2/health/ready endpoint:

```bash
curl -v localhost:8000/v2/health/ready
```

If the server is ready, this command will return a 200 OK response.
