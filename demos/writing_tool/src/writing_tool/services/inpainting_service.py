"""Inpainting service using Hugging Face diffusers model."""

import io
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting
from diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
from PIL import Image

from ..entities.bounding_box import BoundingBox


class InpaintingService:
    """Service for AI-powered image inpainting using Hugging Face diffusers."""

    def __init__(
        self, model_name: str = "Lykon/dreamshaper-8-inpainting", device: str = "cuda"
    ):
        """Initialize the inpainting service.

        Args:
            model_name: Hugging Face model name for inpainting
            device: Device to run the model on (cuda/cpu)
        """
        self.device = device
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None,
        )
        self.pipe.scheduler = DEISMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe = self.pipe.to(device)

    def inpaint(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray],
        prompt: str,
        size: Optional[tuple[int, int]] = None,
        num_inference_steps: int = 25,
        generator_seed: Optional[int] = None,
    ) -> Image.Image:
        """Inpaint an image using diffusers inpainting pipeline.

        Args:
            image: Original image (PIL Image or numpy array)
            mask: Mask defining areas to edit (PIL Image or numpy array)
            prompt: Text description for the edit
            size: Output image size (width, height). If None, uses original size
            num_inference_steps: Number of denoising steps
            generator_seed: Random seed for reproducible results

        Returns:
            Edited image as PIL Image
        """
        # Convert inputs to PIL Images if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(mask, np.ndarray):
            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255
            # For diffusers, mask should be 3-channel with object pixels = 255
            if len(mask.shape) == 2:
                # Convert grayscale to RGB
                mask = np.stack([mask, mask, mask], axis=-1)
            mask = Image.fromarray(mask)

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Ensure mask is RGB (not RGBA for diffusers)
        if mask.mode != "RGB":
            mask = mask.convert("RGB")

        # Resize if size is specified
        if size:
            image = image.resize(size)
            mask = mask.resize(size)

        # Set up generator if seed is provided
        generator = None
        if generator_seed is not None:
            generator = torch.manual_seed(generator_seed)

        # Generate inpainted image
        result = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        return result.images[0]

    def inpaint_multiple(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray],
        prompts: list[str],
        size: Optional[tuple[int, int]] = None,
        num_inference_steps: int = 25,
    ) -> list[Image.Image]:
        """Generate multiple inpainted versions with different prompts.

        Args:
            image: Original image
            mask: Mask defining areas to edit
            prompts: List of text descriptions for different edits
            size: Output image size (width, height)
            num_inference_steps: Number of denoising steps

        Returns:
            List of edited images
        """
        results = []
        for prompt in prompts:
            edited_image = self.inpaint(image, mask, prompt, size, num_inference_steps)
            results.append(edited_image)
        return results

    def create_mask_for_diffusers(
        self, combined_mask: np.ndarray, image_shape: tuple
    ) -> np.ndarray:
        """Create a mask for diffusers inpainting (3-channel with object pixels = 255).

        Args:
            combined_mask: Binary mask (True for areas to edit)
            image_shape: Shape of the original image (height, width, channels)

        Returns:
            RGB mask array with 255 values where inpainting should occur
        """
        if combined_mask.size == 0:
            # Return empty mask (no inpainting)
            rgb_mask = np.zeros((*image_shape[:2], 3), dtype=np.uint8)
            return rgb_mask

        # Create RGB mask: 255 where mask is True (areas to inpaint), 0 elsewhere
        mask_values = combined_mask.astype(np.uint8) * 255
        rgb_mask = np.stack([mask_values, mask_values, mask_values], axis=-1)

        return rgb_mask

    def _image_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Convert PIL Image to bytes."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return buffer.getvalue()

    def create_mask_from_boxes(
        self, boxes: list[BoundingBox], image_shape: tuple
    ) -> np.ndarray:
        """Create a mask from bounding boxes.

        Args:
            boxes: List of BoundingBox objects
            image_shape: Shape of the original image (height, width)

        Returns:
            Mask as numpy array
        """
        mask = np.zeros(image_shape, dtype=np.uint8)

        for box in boxes:
            x_min, y_min, x_max, y_max = box.xyxy
            # Ensure coordinates are within image bounds
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(image_shape[1] - 1, int(x_max))
            y_max = min(image_shape[0] - 1, int(y_max))
            # Fill the mask for the bounding box area
            mask[y_min : y_max + 1, x_min : x_max + 1] = 255

        return mask
