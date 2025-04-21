import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
from skimage.transform import resize
import cv2
from pathlib import Path
import os

from .symbol_detector import SymbolDetector
from ..utilities import Observable


class DiagramProcessor:
    def __init__(self, database):
        self.database = database
        self.detector = SymbolDetector(database)

    def process_user_diagram(self, filename):
        """Process a diagram file located in the same directory as this script"""
        # Get the directory where this Python file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Create the full path to the diagram file
        diagram_path = os.path.join(current_dir, filename)

        # Rest of your processing code
        if not os.path.exists(diagram_path):
            raise FileNotFoundError(f"Diagram file not found: {diagram_path}")

        image = cv2.imread(diagram_path)
        if image is None:
            raise ValueError("Could not read image file")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.detector.update_training_data()
        return self.detector.detect_symbols(image)