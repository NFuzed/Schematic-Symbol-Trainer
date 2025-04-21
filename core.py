from core import *
from src.model import *
from src.core import Database
from src.model3 import *
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


class Core:
    def __init__(self):
        self.database = Database()

        self.symbol_detector = SymbolDetector()

    def display_results(self, image_path):
        # Load models and detect
        detections = []
        for manager in self.database.get_entity_managers():
            dataset = SyntheticSymbolDataset(manager)
            self.symbol_detector.train(dataset)
            detections += self.symbol_detector.detect(image_path)

        # Display the image and draw rectangles
        fig, ax = plt.subplots(1)
        ax.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR), cmap='gray')

        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            w = x2 - x1
            h = y2 - y1
            label = detection['label']
            score = detection['score']

            rect = patches.Rectangle((x1, y1), w, h,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{label} ({score:.2f})", color='red', fontsize=8)

        plt.title("Detected Symbols")
        plt.axis('off')
        plt.show()
