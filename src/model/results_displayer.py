import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import cv2

class ResultsDisplayer:
    def visualize_detections(self, diagram_image, detections, show_confidence=True):
        """
        Draw bounding boxes and labels on the diagram image.

        Args:
            diagram_image: The original diagram image (numpy array)
            detections: List of detection dictionaries from SymbolDetector
            show_confidence: Whether to show confidence scores in labels

        Returns:
            A new image with bounding boxes and labels drawn on it
        """
        diagram_image = cv2.cvtColor(cv2.imread(str(Path(__file__).parent / "diagram.jpg")), cv2.COLOR_BGR2RGB)
        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(12, 12))

        # Display the image
        ax.imshow(diagram_image)

        # Draw each detection
        for detection in detections:
            x, y, w, h = detection['bounding_box']
            label = detection['symbol_type']

            # Create a rectangle patch
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )

            # Add the rectangle to the axes
            ax.add_patch(rect)

            # Add label text
            if show_confidence:
                label_text = f"{label} ({detection['confidence']:.2f})"
            else:
                label_text = label

            ax.text(
                x, y - 10,
                label_text,
                color='red',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8)
            )

        plt.axis('off')
        plt.savefig("output.png", bbox_inches='tight', pad_inches=0, dpi=300)
        return fig
