"""Visualization service for plotting and displaying results."""

import random
from typing import Dict, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from ..entities.detection_result import DetectionResult


class VisualizationService:
    """Service for visualizing detection results and images."""

    def __init__(self):
        """Initialize the visualization service."""
        pass

    def plot_detections_matplotlib(
        self,
        image: Union[Image.Image, np.ndarray],
        detections: List[DetectionResult],
        save_path: Optional[str] = None,
        show_masks: bool = True,
        show_boxes: bool = True,
    ) -> None:
        """Plot detections using matplotlib.

        Args:
            image: Image to display
            detections: List of detection results
            save_path: Optional path to save the plot
            show_masks: Whether to show segmentation masks
            show_boxes: Whether to show bounding boxes
        """
        annotated_image = self.annotate_image(image, detections, show_masks, show_boxes)

        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title(f"Detection Results ({len(detections)} objects found)")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.show()

    def plot_detections_plotly(
        self,
        image: np.ndarray,
        detections: List[DetectionResult],
        class_colors: Optional[Dict[str, str]] = None,
    ) -> None:
        """Plot detections using interactive Plotly visualization.

        Args:
            image: Image array to display
            detections: List of detection results
            class_colors: Optional color mapping for classes
        """
        # Generate colors if not provided
        if class_colors is None:
            colors = self._random_named_css_colors(len(detections))
            class_colors = {str(i): colors[i] for i in range(len(detections))}

        fig = px.imshow(image)

        # Add detection traces
        shapes = []
        annotations = []

        for idx, detection in enumerate(detections):
            label = detection.label
            box = detection.box
            score = detection.score
            mask = detection.mask

            # Add mask polygon if available
            if mask is not None:
                polygon = self._mask_to_polygon(mask)
                if polygon:
                    fig.add_trace(
                        go.Scatter(
                            x=[point[0] for point in polygon] + [polygon[0][0]],
                            y=[point[1] for point in polygon] + [polygon[0][1]],
                            mode="lines",
                            line=dict(color=class_colors[str(idx)], width=2),
                            fill="toself",
                            fillcolor=f"rgba{self._hex_to_rgba(class_colors[str(idx)], 0.3)}",
                            name=f"{label}: {score:.2f}",
                            hovertemplate=f"<b>{label}</b><br>Confidence: {score:.2f}<extra></extra>",
                        )
                    )

            # Add bounding box
            xmin, ymin, xmax, ymax = box.xyxy
            shape = dict(
                type="rect",
                xref="x",
                yref="y",
                x0=xmin,
                y0=ymin,
                x1=xmax,
                y1=ymax,
                line=dict(color=class_colors[str(idx)], width=2),
                fillcolor=f"rgba{self._hex_to_rgba(class_colors[str(idx)], 0.1)}",
            )

            annotation = dict(
                x=(xmin + xmax) / 2,
                y=ymin - 10,
                xref="x",
                yref="y",
                text=f"{label}: {score:.2f}",
                showarrow=False,
                bgcolor=class_colors[str(idx)],
                bordercolor="white",
                borderwidth=1,
            )

            shapes.append(shape)
            annotations.append(annotation)

        # Configure interactive buttons
        button_shapes = [dict(label="None", method="relayout", args=["shapes", []])]
        button_shapes.extend(
            [
                dict(
                    label=f"Detection {idx+1}",
                    method="relayout",
                    args=["shapes", [shape]],
                )
                for idx, shape in enumerate(shapes)
            ]
        )
        button_shapes.append(
            dict(label="All", method="relayout", args=["shapes", shapes])
        )

        # Update layout
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=True,
            title=f"Interactive Detection Results ({len(detections)} objects)",
            updatemenus=[
                dict(
                    type="buttons",
                    direction="up",
                    x=0.01,
                    y=0.99,
                    buttons=button_shapes,
                )
            ],
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            shapes=shapes,
            annotations=annotations,
        )

        fig.show()

    def annotate_image(
        self,
        image: Union[Image.Image, np.ndarray],
        detection_results: List[DetectionResult],
        show_masks: bool = True,
        show_boxes: bool = True,
    ) -> np.ndarray:
        """Annotate image with detection results.

        Args:
            image: Input image
            detection_results: List of detection results
            show_masks: Whether to show segmentation masks
            show_boxes: Whether to show bounding boxes

        Returns:
            Annotated image as numpy array
        """
        # Convert PIL Image to OpenCV format
        image_cv2 = np.array(image) if isinstance(image, Image.Image) else image.copy()
        if len(image_cv2.shape) == 3:
            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

        # Iterate over detections and add annotations
        for detection in detection_results:
            label = detection.label
            score = detection.score
            box = detection.box
            mask = detection.mask

            # Sample a random color for each detection
            color = np.random.randint(0, 256, size=3).tolist()

            # Draw bounding box
            if show_boxes:
                cv2.rectangle(
                    image_cv2,
                    (int(box.xmin), int(box.ymin)),
                    (int(box.xmax), int(box.ymax)),
                    color,
                    2,
                )
                cv2.putText(
                    image_cv2,
                    f"{label}: {score:.2f}",
                    (int(box.xmin), int(box.ymin) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            # Draw mask contours
            if show_masks and mask is not None:
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(image_cv2, contours, -1, color, 2)

        # Convert back to RGB
        if len(image_cv2.shape) == 3:
            return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        return image_cv2

    def display_mask(self, mask: np.ndarray, title: str = "Segmentation Mask") -> None:
        """Display a segmentation mask.

        Args:
            mask: Binary mask to display
            title: Title for the plot
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(mask, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()

    def compare_images(
        self,
        original: Union[Image.Image, np.ndarray],
        edited: Union[Image.Image, np.ndarray],
        titles: Optional[List[str]] = None,
    ) -> None:
        """Display original and edited images side by side.

        Args:
            original: Original image
            edited: Edited image
            titles: Optional titles for the images
        """
        if titles is None:
            titles = ["Original Image", "Edited Image"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(original)
        axes[0].set_title(titles[0])
        axes[0].axis("off")

        axes[1].imshow(edited)
        axes[1].set_title(titles[1])
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    def _random_named_css_colors(self, num_colors: int) -> List[str]:
        """Generate random CSS color names."""
        named_css_colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "navy",
            "lime",
            "aqua",
            "teal",
            "silver",
            "maroon",
            "yellow",
            "fuchsia",
            "white",
            "black",
            "coral",
            "crimson",
            "gold",
            "indigo",
            "violet",
            "turquoise",
            "plum",
            "salmon",
            "khaki",
            "tan",
            "orchid",
        ]
        return random.sample(named_css_colors, min(num_colors, len(named_css_colors)))

    def _hex_to_rgba(self, hex_color: str, alpha: float = 1.0) -> str:
        """Convert hex color to RGBA string."""
        # Simple implementation for common color names
        color_map = {
            "red": "(255,0,0",
            "blue": "(0,0,255",
            "green": "(0,128,0",
            "orange": "(255,165,0",
            "purple": "(128,0,128",
        }
        base = color_map.get(hex_color, "(128,128,128")
        return f"{base},{alpha})"

    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        """Convert mask to polygon coordinates."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the vertices of the contour
        polygon = largest_contour.reshape(-1, 2).tolist()

        return polygon
