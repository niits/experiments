"""Segmentation service using Segment Anything Model (SAM)."""

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor

from ..entities.bounding_box import BoundingBox
from ..entities.detection_result import DetectionResult
from ..logging_config import get_logger


class SegmentationService:
    """Service for object segmentation using Segment Anything Model."""

    def __init__(self, model_id: Optional[str] = None):
        """Initialize the segmentation service.

        Args:
            model_id: Model ID for SAM (default: facebook/sam-vit-base)
        """
        self.logger = get_logger(__name__)
        self.model_id = model_id or "facebook/sam-vit-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """Load the SAM model and processor."""
        try:
            self.logger.info(f"Loading SAM model: {self.model_id}")
            self._processor = None  # Lazy load the processor
            self._segmentator = None  # Lazy load the segmentator
            self.model = AutoModelForMaskGeneration.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.logger.info(f"SAM model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load SAM model: {e}")
            raise

    @property
    def segmentator(self):
        """Lazy load the segmentation model."""
        if self._segmentator is None:
            self._segmentator = AutoModelForMaskGeneration.from_pretrained(
                self.model_id
            ).to(self.device)
        return self._segmentator

    @property
    def processor(self):
        """Lazy load the processor."""
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        return self._processor

    def segment(
        self,
        image: Image.Image,
        detection_results: List[DetectionResult],
        polygon_refinement: bool = False,
    ) -> List[DetectionResult]:
        """Generate segmentation masks for detected objects.

        Args:
            image: PIL Image to segment
            detection_results: List of DetectionResult objects with bounding boxes
            polygon_refinement: Whether to refine masks using polygon approximation

        Returns:
            List of DetectionResult objects with masks added
        """
        if not detection_results:
            return detection_results

        # Extract bounding boxes
        boxes = self._get_boxes(detection_results)

        # Process inputs
        inputs = self.processor(
            images=image, input_boxes=boxes, return_tensors="pt"
        ).to(self.device)

        # Generate masks
        with torch.no_grad():
            outputs = self.segmentator(**inputs)

        # Post-process masks
        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )[0]

        # Refine masks
        refined_masks = self._refine_masks(masks, polygon_refinement)

        # Add masks to detection results
        for detection_result, mask in zip(detection_results, refined_masks):
            detection_result.mask = mask

        return detection_results

    def _get_boxes(self, results: List[DetectionResult]) -> List[List[List[float]]]:
        """Extract bounding boxes from detection results."""
        boxes = []
        for result in results:
            xyxy = result.box.xyxy
            boxes.append(xyxy)
        return [boxes]

    def _refine_masks(
        self, masks: torch.BoolTensor, polygon_refinement: bool = False
    ) -> List[np.ndarray]:
        """Refine and convert masks to numpy arrays."""
        # Convert boolean tensor to numpy arrays
        masks_np = masks.cpu().numpy().astype(np.uint8)

        # Convert from tensor format (N, C, H, W) to list of 2D arrays (H, W)
        if masks_np.ndim == 4:
            # Handle both (N, 1, H, W) and (N, 3, H, W) cases
            if masks_np.shape[1] == 1:
                masks_np = masks_np.squeeze(1)  # Remove channel dimension
            elif masks_np.shape[1] == 3:
                # Take the first channel or combine channels
                masks_np = masks_np[:, 0, :, :]  # Take first channel
        elif masks_np.ndim == 3:
            # Handle case where we have (C, H, W) for single mask
            if masks_np.shape[0] == 3:
                masks_np = masks_np[0, :, :]  # Take first channel
                masks_np = masks_np[np.newaxis, ...]  # Add batch dimension back

        # Convert to list of individual masks
        mask_list: List[np.ndarray] = []
        for i in range(masks_np.shape[0]):
            mask_list.append(masks_np[i])

        if polygon_refinement:
            for idx, mask in enumerate(mask_list):
                try:
                    shape = mask.shape
                    polygon = self._mask_to_polygon(mask)
                    if polygon:  # Only refine if we got a valid polygon
                        refined_mask = self._polygon_to_mask(polygon, shape)
                        mask_list[idx] = refined_mask
                    # If polygon is empty, keep original mask
                except Exception as e:
                    # If refinement fails, keep the original mask
                    self.logger.warning(f"Mask refinement failed for mask {idx}: {e}")
                    continue

        return mask_list

    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        """Convert mask to polygon coordinates."""
        # Ensure mask is 2D and uint8
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        if len(mask.shape) > 2:
            mask = mask[0]  # Take first channel if still multi-dimensional

        # Ensure mask is in valid range and type
        mask = np.clip(mask, 0, 1)
        mask = (mask * 255).astype(np.uint8)

        # Check if mask has valid dimensions
        if mask.shape[0] == 0 or mask.shape[1] == 0:
            return []

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the vertices of the contour and convert to int
        polygon = largest_contour.reshape(-1, 2).astype(int).tolist()

        return polygon

    def _polygon_to_mask(
        self, polygon: List[List[int]], image_shape: tuple
    ) -> np.ndarray:
        """Convert polygon to segmentation mask."""
        mask = np.zeros(image_shape, dtype=np.uint8)

        if not polygon:
            return mask

        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=255)

        return mask

    def combine_masks(self, detection_results: List[DetectionResult]) -> np.ndarray:
        """Combine multiple masks into a single mask using logical OR.

        Args:
            detection_results: List of DetectionResult objects with masks

        Returns:
            Combined mask as numpy array
        """
        masks = [
            detection.mask
            for detection in detection_results
            if detection.mask is not None
        ]

        if not masks:
            return np.array([])

        combined_mask = masks[0].astype(bool)
        for mask in masks[1:]:
            combined_mask = np.logical_or(combined_mask, mask.astype(bool))

        return combined_mask
