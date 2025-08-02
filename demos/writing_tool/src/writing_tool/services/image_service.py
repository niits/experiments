"""Image service for loading and processing images."""

from typing import Union

import numpy as np
import requests
from PIL import Image


class ImageService:
    """Service for image loading and basic processing operations."""

    def __init__(self):
        """Initialize the image service."""
        pass

    def load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """Load an image from various sources.

        Args:
            image_source: Image source (URL, file path, or PIL Image)

        Returns:
            PIL Image in RGB format
        """
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")

        if isinstance(image_source, str):
            if image_source.startswith("http"):
                # Load from URL
                response = requests.get(image_source, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw)
                if hasattr(response.raw, "decode_content"):
                    response.raw.decode_content = True
                image = image.convert("RGB")
            else:
                # Load from file path
                image = Image.open(image_source).convert("RGB")
            return image

        raise ValueError("Unsupported image source type")

    def resize_image(
        self,
        image: Image.Image,
        size: tuple[int, int],
        maintain_aspect_ratio: bool = True,
    ) -> Image.Image:
        """Resize an image.

        Args:
            image: PIL Image to resize
            size: Target size as (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio

        Returns:
            Resized PIL Image
        """
        if maintain_aspect_ratio:
            image.thumbnail(size, Image.Resampling.LANCZOS)
            return image
        else:
            return image.resize(size, Image.Resampling.LANCZOS)

    def crop_image(
        self, image: Image.Image, box: tuple[int, int, int, int]
    ) -> Image.Image:
        """Crop an image.

        Args:
            image: PIL Image to crop
            box: Crop box as (left, top, right, bottom)

        Returns:
            Cropped PIL Image
        """
        return image.crop(box)

    def to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array.

        Args:
            image: PIL Image to convert

        Returns:
            Image as numpy array
        """
        return np.array(image)

    def from_numpy(self, array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image.

        Args:
            array: Numpy array to convert

        Returns:
            PIL Image
        """
        return Image.fromarray(array)

    def create_transparency_overlay(
        self, image: Image.Image, mask: np.ndarray
    ) -> Image.Image:
        """Create an image with transparency overlay based on mask.

        Args:
            image: Original image
            mask: Binary mask (True for transparent areas)

        Returns:
            Image with transparency overlay as RGBA
        """
        # Convert image to RGBA
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        image_array = np.array(image)

        if mask.size > 0:
            # Apply transparency where mask is True
            alpha_channel = (1 - mask).astype(np.uint8) * 255
            image_array[:, :, 3] = alpha_channel

        return Image.fromarray(image_array, mode="RGBA")

    def save_image(
        self, image: Image.Image, path: str, format: str = "PNG", quality: int = 95
    ) -> None:
        """Save an image to file.

        Args:
            image: PIL Image to save
            path: File path to save to
            format: Image format (PNG, JPEG, etc.)
            quality: JPEG quality (0-100)
        """
        if format.upper() == "JPEG":
            # Convert RGBA to RGB for JPEG
            if image.mode == "RGBA":
                # Create white background
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(
                    image, mask=image.split()[-1]
                )  # Use alpha channel as mask
                image = background
            image.save(path, format=format, quality=quality)
        else:
            image.save(path, format=format)

    def get_image_info(self, image: Image.Image) -> dict:
        """Get information about an image.

        Args:
            image: PIL Image to analyze

        Returns:
            Dictionary with image information
        """
        return {
            "size": image.size,
            "mode": image.mode,
            "format": image.format,
            "width": image.width,
            "height": image.height,
        }
