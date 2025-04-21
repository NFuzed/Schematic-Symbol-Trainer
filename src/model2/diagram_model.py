import cv2
from .entity_detector import EntityDetector
import numpy as np


class DiagramModel:
    def __init__(self, detector: EntityDetector):
        self.detector = detector
        self.diagram_image = None
        self.entities_detected = []

    def load_diagram(self, image_path):
        self.diagram_image = cv2.imread(image_path)

    def detect_entities(self):
        if self.diagram_image is None:
            raise ValueError("No diagram image loaded.")
        self.entities_detected = self.detector.detect_entities(self.diagram_image)

    def get_detected_entities(self):
        return self.entities_detected

    # NEW METHOD TO DISPLAY RESULTS
    def display_detections(self, output_path=None, display_window=True):
        if self.diagram_image is None:
            raise ValueError("No diagram image loaded.")

        annotated_image = self.diagram_image.copy()

        for entity in self.entities_detected:
            x, y = entity['position']
            w, h = entity['size']
            entity_type = entity['entity_type']

            # Draw rectangle around detected entity
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add label above rectangle
            cv2.putText(
                annotated_image,
                entity_type,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        # Optionally save the annotated image
        if output_path:
            cv2.imwrite(output_path, annotated_image)

        # Optionally display the annotated image in a window
        if display_window:
            cv2.imshow("Detected Entities", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return annotated_image