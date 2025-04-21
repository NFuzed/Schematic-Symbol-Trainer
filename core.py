from core import *
from src.model import *
from src.core import Database
from src.model import *
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


class Core:
    def __init__(self):
        self.database = Database()
        self.symbol_detector = SymbolDetector()

    def display_results(self, image_path):
        entity_managers = sorted(self.database.get_entity_managers(), key=lambda m: m.entity_manager_name)
        dataset = MultiClassSymbolDataset(entity_managers)
        self.symbol_detector.train(dataset, num_classes=len(entity_managers) + 1)
        detections = self.symbol_detector.detect(image_path)

        label_to_name = {i + 1: mgr.entity_manager_name for i, mgr in enumerate(entity_managers)}

        fig, ax = plt.subplots(1)
        ax.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR), cmap='gray')

        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            w = x2 - x1
            h = y2 - y1
            label = detection['label']
            score = detection['score']
            name = label_to_name.get(label, f"Class {label}")

            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{name} ({score:.2f})", color='red', fontsize=8)

        plt.title("Detected Symbols")
        plt.axis('off')
        plt.show()
