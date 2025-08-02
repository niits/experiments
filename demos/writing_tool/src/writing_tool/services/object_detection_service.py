"""Object detection service using GroundingDINO."""

from typing import List, Optional

import torch
from PIL import Image
from transformers import pipeline

from ..entities.detection_result import DetectionResult


class ObjectDetectionService:
    """Service for object detection using GroundingDINO."""

    def __init__(self, model_id: Optional[str] = None):
        """Initialize the object detection service.

        Args:
            model_id: The model ID for GroundingDINO. Defaults to IDEA-Research/grounding-dino-tiny.
        """
        self.model_id = model_id or "IDEA-Research/grounding-dino-tiny"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._detector = None

    @property
    def detector(self):
        """Lazy load the detector to avoid loading it until needed."""
        if self._detector is None:
            self._detector = pipeline(
                model=self.model_id,
                task="zero-shot-object-detection",
                device=self.device,
            )
        return self._detector

    def detect(
        self, image: Image.Image, labels: List[str], threshold: float = 0.3
    ) -> List[DetectionResult]:
        """Detect objects in an image using text prompts.

        Args:
            image: PIL Image to detect objects in
            labels: List of text labels to detect (e.g., ["cat", "dog"])
            threshold: Confidence threshold for detections

        Returns:
            List of DetectionResult objects containing bounding boxes and metadata
        """
        # Ensure labels end with periods for better detection
        formatted_labels = [
            label if label.endswith(".") else label + "." for label in labels
        ]

        # Run detection
        raw_results = self.detector(
            image, candidate_labels=formatted_labels, threshold=threshold
        )

        # Convert to DetectionResult objects
        detections = [DetectionResult.from_dict(result) for result in raw_results]

        return detections
