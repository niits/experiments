# Writing tool

A powerful AI-powered image editing tool that helps improve writing skills by allowing users to edit images based on text descriptions. The tool uses GroundingDINO for object detection, Segment Anything Model (SAM) for segmentation, and Hugging Face's diffusers library for **completely local AI-powered image inpainting** (no API keys required).

## Architecture Overview

The package has been refactored from a monolithic utility-based approach to a clean service-oriented architecture:

### Directory Structure

```

src/writing_tool/
├── __init__.py                 # Main package exports
├── entities/                   # Data models
│   ├── __init__.py
│   ├── bounding_box.py        # BoundingBox entity
│   └── detection_result.py    # DetectionResult entity
├── services/                   # Business logic services
│   ├── __init__.py
│   ├── object_detection_service.py    # GroundingDINO detection
│   ├── segmentation_service.py        # SAM segmentation
│   ├── prompt_normalization_service.py # Text prompt processing
│   ├── inpainting_service.py          # Hugging Face diffusers inpainting
│   ├── visualization_service.py       # Plotting and display
│   ├── image_service.py               # Image loading/processing
│   └── writing_tool_service.py        # Main orchestrator
└── utils.py                    # Legacy utilities (backward compatibility)

```

## Services Overview

### 1. ObjectDetectionService

Handles object detection using GroundingDINO:

- Zero-shot object detection with text prompts
- Configurable confidence thresholds
- Support for multiple detection models

### 2. SegmentationService

Manages segmentation using Segment Anything Model:

- Precise mask generation from bounding boxes
- Polygon refinement options
- Mask combination utilities

### 3. PromptNormalizationService

Processes and improves text prompts:

- Multiple writing styles (simple, detailed, narrative, artistic)
- Prompt variations generation
- Text analysis and key element extraction

### 4. InpaintingService

Handles AI-powered image editing:

- Hugging Face diffusers integration with DreamShaper-8
- Multiple output sizes
- Local GPU processing (no API keys required)

### 5. VisualizationService

Provides visualization utilities:

- Interactive Plotly plots
- Matplotlib static plots
- Image comparison views

### 6. ImageService

Manages image operations:

- Loading from URLs and files
- Format conversions
- Basic image processing

### 7. WritingToolService (Main Orchestrator)

Coordinates all services for complete workflows:

- End-to-end image editing pipeline
- Individual service access  
- Configurable processing options
- New `edit_image_end_to_end()` method for simplified workflows

## Quick Start

### Installation

```bash
# Install the package and dependencies
uv sync --index pytorch-cpu  # For CPU
# or
uv sync --index pytorch-cu121  # For CUDA 12.1
```

### Basic Usage

```python
from writing_tool.services import WritingToolService
import os
from dotenv import load_dotenv

# Initialize the main service
writing_tool = WritingToolService(
    detector_model_id="IDEA-Research/grounding-dino-tiny",
    segmenter_model_id="facebook/sam-vit-base"
)

# Complete workflow
results = writing_tool.process_image_with_text(
    image_source="path/to/image.jpg",
    text_description="Two cats sleeping on a sofa",
    detection_labels=["cat", "sofa"],
    prompt_style="detailed",
    output_size="512x512"
)

# Access results
edited_image = results['edited_image']
detections = results['detections']
normalized_prompt = results['normalized_prompt']
```

### End-to-End Workflow

For simplified usage, use the new end-to-end function:

```python
# Single function call for complete workflow
edited_image = writing_tool.edit_image_end_to_end(
    image_source="path/to/image.jpg",
    text_description="Two cats sleeping on a sofa",
    detection_labels=["cat", "sofa"],
    detection_threshold=0.3,
    output_size="512x512",
    generator_seed=42  # For reproducible results
)

# edited_image is a PIL Image ready to use
edited_image.save("edited_result.jpg")
```

### Individual Services

```python
# Use individual services for fine-grained control
image = writing_tool.image_service.load_image("image.jpg")
detections = writing_tool.detection_service.detect(image, ["cat", "dog"])
detections_with_masks = writing_tool.segmentation_service.segment(image, detections)

# Normalize prompts
prompt = writing_tool.prompt_service.normalize_prompt(
    "cats on sofa", 
    style="artistic"
)

# Visualize results
writing_tool.visualization_service.plot_detections_plotly(
    image_array, detections_with_masks
)
```

## Examples

### 1. Jupyter Notebook

See `main.ipynb` for an interactive example that demonstrates:

- Step-by-step workflow
- Individual service usage
- Visualization options
- Complete integration example

## Use Cases

### 1. Writing Enhancement

- Edit images to match written descriptions
- Generate visual content for stories
- Create consistent imagery for articles

## Development

### Adding New Services

1. Create service class in `src/writing_tool/services/`
2. Follow the established patterns
3. Add to `services/__init__.py`
4. Update main package exports

## License

This project maintains the same license as the original implementation.
