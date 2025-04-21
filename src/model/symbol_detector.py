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

class SymbolDetector:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = T.Compose([T.ToTensor()])
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item():.4f}")

        torch.save(self.model.state_dict(), os.path.join(self.model_dir, "symbol_detector.pth"))

    def load_model(self, num_classes):
        self.model = self._create_model(num_classes)
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, "symbol_detector.pth"), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def detect(self, diagram_path, score_threshold=0.5):
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
        for i in range(min(10, len(temp_scores))):
            print(temp_scores[i])

        return detections

class MultiClassSymbolDataset(Dataset):
    def __init__(self, entity_managers: list[EntityManager], image_size=640, samples_per_image=3, transform=None, background_color=(255, 255, 255)):
        self.entity_managers = sorted(entity_managers, key=lambda m: m.entity_manager_name)
        self.image_size = image_size
        self.samples_per_image = samples_per_image
        self.transform = transform if transform else T.ToTensor()
        self.background_color = background_color

        self.label_to_entities = {i + 1: mgr.entities for i, mgr in enumerate(self.entity_managers)}
        self.label_to_names = {i + 1: mgr.entity_manager_name for i, mgr in enumerate(self.entity_managers)}
        self.label_list = list(self.label_to_entities.keys())
        self.all_samples = [(label, entity) for label, entities in self.label_to_entities.items() for entity in entities]

    def __len__(self):
        return len(self.all_samples) + len(self.all_samples) // 4  # +25% background-only samples

    def __getitem__(self, idx):
        bg_np = np.full((self.image_size, self.image_size, 3), self.background_color, dtype=np.uint8)
        boxes = []
        labels = []

        if idx >= len(self.all_samples):  # background-only image
            bg_tensor = self.transform(bg_np)
            return bg_tensor, {"boxes": torch.zeros((0, 4), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.int64)}

        # Otherwise, generate symbol-filled image
        for _ in range(self.samples_per_image):
            label, entity = random.choice(self.all_samples)
            symbol = Image.fromarray(entity.image).convert("RGB").resize((32, 32))

            # Augmentations
            if random.random() > 0.5:
                symbol = symbol.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                symbol = symbol.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                angle = random.randint(-20, 20)
                symbol = symbol.rotate(angle, expand=True)
            if random.random() > 0.5:
                symbol = symbol.filter(ImageFilter.GaussianBlur(radius=1))
            if random.random() > 0.5:
                enhancer = ImageEnhance.Contrast(symbol)
                symbol = enhancer.enhance(random.uniform(0.8, 1.2))

            sx, sy = symbol.size
            x = random.randint(4, self.image_size - sx - 4)
            y = random.randint(4, self.image_size - sy - 4)
            bg_np[y:y+sy, x:x+sx] = np.array(symbol)
            boxes.append([x, y, x+sx, y+sy])
            labels.append(label)

        bg_tensor = self.transform(bg_np)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return bg_tensor, target