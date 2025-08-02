"""Writing Tool Package - AI-powered image editing for writing improvement."""

from .entities import BoundingBox, DetectionResult
from .services import (
    ImageService,
    InpaintingService,
    ObjectDetectionService,
    PromptNormalizationService,
    SegmentationService,
    VisualizationService,
    WritingToolService,
)

__version__ = "1.0.0"


def main() -> None:
    from .logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Hello from writing-tool!")


__all__ = [
    # Entities
    "BoundingBox",
    "DetectionResult",
    # Services
    "ObjectDetectionService",
    "SegmentationService",
    "PromptNormalizationService",
    "InpaintingService",
    "VisualizationService",
    "ImageService",
    "WritingToolService",
    # Main function
    "main",
]
