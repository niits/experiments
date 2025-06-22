SAVE_DIR := $(CURDIR)/tensorrt_engines

ONNX_DIR := /mnt/models/onnx/stable-diffusion-v1-5

setup-bash:
	set -eou pipefail;

create-engines-dir: setup-bash
	mkdir -p $(SAVE_DIR)

build-tensorrt-for-text-encoder: create-engines-dir

	echo "Building TensorRT engine for text encoder";
	mkdir -p $(SAVE_DIR)/text_encoder;	
	
	trtexec --onnx=$(ONNX_DIR)/text_encoder/model.onnx \
		--minShapes=input_ids:1x77 \
		--optShapes=input_ids:2x77 \
		--maxShapes=input_ids:8x77 \
		--shapes=input_ids:2x77 \
		--saveEngine=${SAVE_DIR}/text_encoder/model.plan \
		--workspace=600;


build-tensorrt-for-vae-decoder: create-engines-dir

	echo "Building TensorRT engine for vae decoder";
	mkdir -p $(SAVE_DIR)/vae_decoder;	
	
	trtexec --onnx=$(ONNX_DIR)/vae_decoder/model.onnx \
		--minShapes=latent_sample:1x4x64x64 \
		--optShapes=latent_sample:2x4x64x64 \
		--maxShapes=latent_sample:4x4x64x64 \
		--shapes=latent_sample:2x4x64x64 \
		--saveEngine=$(SAVE_DIR)/vae_decoder/model.plan \
		--workspace=8000 \
		--verbose;

build-tensorrt-for-unet: create-engines-dir

	echo "Building TensorRT engine for unet";
	mkdir -p $(SAVE_DIR)/unet;	
	
	trtexec --onnx=$(ONNX_DIR)/unet/model.onnx \
		--minShapes=sample:1x4x64x64,timestep:1x1,encoder_hidden_states:1x77x768 \
		--optShapes=sample:2x4x64x64,timestep:2x1,encoder_hidden_states:2x77x768 \
		--maxShapes=sample:4x4x64x64,timestep:4x1,encoder_hidden_states:4x77x768 \
		--shapes=sample:2x4x64x64,timestep:2x1,encoder_hidden_states:2x77x768 \
		--saveEngine=tensorrt_engines/unet.engine \
		--verbose  \
		--workspace=16000;

all: build-tensorrt-for-text-encoder build-tensorrt-for-vae-decoder build-tensorrt-for-unet