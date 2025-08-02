"""Main writing tool service that orchestrates all components."""

from typing import List, Literal, Optional, Union

import numpy as np
from PIL import Image

from ..entities.detection_result import DetectionResult
from ..logging_config import get_logger
from .image_service import ImageService
from .inpainting_service import InpaintingService
from .object_detection_service import ObjectDetectionService
from .prompt_normalization_service import PromptNormalizationService
from .segmentation_service import SegmentationService
from .visualization_service import VisualizationService


class WritingToolService:
    """Main service that orchestrates the writing tool workflow."""

    def __init__(
        self,
        detector_model_id: Optional[str] = None,
        segmenter_model_id: Optional[str] = None,
        inpainting_model_id: str = "Lykon/dreamshaper-8-inpainting",
        device: str = "cuda",
    ):
        """Initialize the writing tool service.

        Args:
            detector_model_id: Model ID for object detection
            segmenter_model_id: Model ID for segmentation
            inpainting_model_id: Hugging Face model ID for inpainting
            device: Device to run models on (cuda/cpu)
        """
        self.logger = get_logger(__name__)
        self.image_service = ImageService()
        self.detection_service = ObjectDetectionService(detector_model_id)
        self.segmentation_service = SegmentationService(segmenter_model_id)
        self.prompt_service = PromptNormalizationService()
        self.inpainting_service = InpaintingService(inpainting_model_id, device)
        self.visualization_service = VisualizationService()

        self.logger.info(f"WritingToolService initialized with device: {device}")
        self.logger.debug(
            f"Services: Detection={detector_model_id}, Segmentation={segmenter_model_id}, Inpainting={inpainting_model_id}"
        )

    def _convert_size_to_tuple(
        self, size: Literal["256x256", "512x512", "1024x1024"]
    ) -> tuple[int, int]:
        """Convert string size format to tuple format.

        Args:
            size: Size in "WxH" format

        Returns:
            Tuple of (width, height)
        """
        width, height = map(int, size.split("x"))
        return (width, height)

    def process_image_with_text(
        self,
        image_source: Union[str, Image.Image],
        text_description: str,
        detection_labels: List[str],
        detection_threshold: float = 0.3,
        polygon_refinement: bool = True,
        prompt_style: str = "detailed",
        output_size: Literal["256x256", "512x512", "1024x1024"] = "512x512",
        visualize_results: bool = True,
    ) -> dict:
        raise NotImplementedError(
            "This method should be implemented to handle the complete image editing workflow."
        )

    def edit_image_end_to_end(
        self,
        image_source: Union[str, Image.Image],
        text_description: str,
        detection_labels: List[str],
        detection_threshold: float = 0.3,
        polygon_refinement: bool = True,
        prompt_style: str = "detailed",
        output_size: Literal["256x256", "512x512", "1024x1024"] = "512x512",
        num_inference_steps: int = 25,
        generator_seed: Optional[int] = None,
    ) -> Image.Image:
        """End-to-end image editing with a single function call.

        Args:
            image_source: Image source (URL, path, or PIL Image)
            text_description: Text description for image editing
            detection_labels: Labels for object detection
            detection_threshold: Confidence threshold for detection
            polygon_refinement: Whether to refine segmentation masks
            prompt_style: Style for prompt normalization
            output_size: Size for generated image
            num_inference_steps: Number of inference steps for diffusion
            generator_seed: Random seed for reproducible results

        Returns:
            Final edited image
        """
        self.logger.info("Starting end-to-end image editing workflow")

        # Use the comprehensive workflow but extract only the final image
        results = self.process_image_with_text(
            image_source=image_source,
            text_description=text_description,
            detection_labels=detection_labels,
            detection_threshold=detection_threshold,
            polygon_refinement=polygon_refinement,
            prompt_style=prompt_style,
            output_size=output_size,
            visualize_results=False,  # No visualization for simplified workflow
        )

        edited_image = results.get("edited_image")
        if edited_image is None:
            self.logger.error("Failed to generate edited image")
            raise RuntimeError("Failed to generate edited image")

        self.logger.info("End-to-end image editing completed successfully")
        return edited_image

    def detect_and_segment(
        self,
        image_source: Union[str, Image.Image],
        labels: List[str],
        threshold: float = 0.3,
        polygon_refinement: bool = True,
        visualize: bool = True,
    ) -> tuple[np.ndarray, List[DetectionResult]]:
        """Detect objects and generate segmentation masks.

        Args:
            image_source: Image source
            labels: Labels for detection
            threshold: Detection threshold
            polygon_refinement: Whether to refine masks
            visualize: Whether to show results

        Returns:
            Tuple of (image_array, detections_with_masks)
        """
        # Load image
        image = self.image_service.load_image(image_source)
        image_array = self.image_service.to_numpy(image)

        # Detect objects
        detections = self.detection_service.detect(image, labels, threshold)

        # Generate masks
        detections_with_masks = self.segmentation_service.segment(
            image, detections, polygon_refinement
        )

        # Visualize if requested
        if visualize:
            self.visualization_service.plot_detections_plotly(
                image_array, detections_with_masks
            )

        return image_array, detections_with_masks

    def generate_prompt_variations(self, base_text: str, count: int = 3) -> List[str]:
        """Generate multiple variations of a text prompt.

        Args:
            base_text: Base text to create variations from
            count: Number of variations to generate

        Returns:
            List of prompt variations
        """
        return self.prompt_service.generate_variations(base_text, count)

    def edit_image_with_prompt(
        self,
        image: Image.Image,
        mask: Union[Image.Image, np.ndarray],
        prompt: str,
        size: Literal["256x256", "512x512", "1024x1024"] = "512x512",
    ) -> Image.Image:
        """Edit an image using a text prompt and mask.

        Args:
            image: Original image
            mask: Editing mask
            prompt: Text description for editing
            size: Output image size

        Returns:
            Edited image
        """
        size_tuple = self._convert_size_to_tuple(size)
        return self.inpainting_service.inpaint(image, mask, prompt, size_tuple)

    def _visualize_workflow_results(self, results: dict) -> None:
        """Visualize the complete workflow results."""
        # Show detection results
        if "detections_with_masks" in results and results["detections_with_masks"]:
            self.visualization_service.plot_detections_plotly(
                results["image_array"], results["detections_with_masks"]
            )

        # Show combined mask
        if "combined_mask" in results and results["combined_mask"].size > 0:
            self.visualization_service.display_mask(
                results["combined_mask"], "Combined Segmentation Mask"
            )

        # Show transparency overlay
        if "transparency_mask" in results:
            self.visualization_service.display_mask(
                np.array(results["transparency_mask"])[:, :, 3],
                "Transparency Mask for Editing",
            )

        # Compare original and edited images
        if "original_image" in results and "edited_image" in results:
            self.visualization_service.compare_images(
                results["original_image"],
                results["edited_image"],
                ["Original Image", "AI-Edited Image"],
            )
