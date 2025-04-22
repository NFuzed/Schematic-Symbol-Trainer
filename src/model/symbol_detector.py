import os
import random
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import draw_bounding_boxes
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from src.core.database import Database
from src.core.entity_manager import EntityManager
from src.utilities.model_results_data_exporter import export_detections_as_csv_and_image
from src.utilities.observable import Observable


class SymbolDetector:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = T.Compose([T.ToTensor()])
        self.update_logger_observer = Observable()
        self.model = None

    def _create_model(self, num_classes):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def train(self, dataset, num_classes, num_epochs=10, lr=0.005):
        self.model = self._create_model(num_classes)
        self.model.to(self.device)

        data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        self.model.train()
        for epoch in range(num_epochs):
            losses = None
            for images, targets in data_loader:
                images = list(img.to(self.device) for img in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            lr_scheduler.step()
            self.update_logger_observer.notify(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item():.4f}")

        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "symbol_detector.pth"))

    def load_model(self, num_classes=2, path=None):
        if path is None:
            path = os.path.join(self.model_dir, "symbol_detector.pth")
        self.model = self._create_model(num_classes)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def detect(self, diagram_path, score_threshold=0.5, export_folder=None):
        if self.model is None:
            raise ValueError("Model must be trained or loaded before detection.")

        self.model.eval()
        image = Image.open(diagram_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()

        detections = []
        temp_scores = []
        for box, score, label in zip(boxes, scores, labels):
            temp_scores.append(score)
            if score >= score_threshold:
                detections.append({
                    'box': box.tolist(),
                    'score': float(score),
                    'label': int(label)
                })

        temp_scores.sort(reverse=True)

        if export_folder is not None:
            export_detections_as_csv_and_image(diagram_path, export_folder, detections)

        return detections

    def export_model(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)