import os
import random
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import numpy as np

from src.core import EntityManager


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

    def  train(self, dataset, num_classes=2, num_epochs=10, lr=0.005):
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

    def load_model(self, num_classes=2):
        self.model = self._create_model(num_classes)
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, "symbol_detector.pth"), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def detect(self, diagram_path, score_threshold=0.9):
        if self.model is None:
            self.load_model()

        self.model.eval()

        image = Image.open(diagram_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()

        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score >= score_threshold:
                detections.append({
                    'box': box.tolist(),
                    'score': float(score),
                    'label': int(label)
                })

        return detections

class SyntheticSymbolDataset(Dataset):
    def __init__(self, entity_manager: EntityManager, image_size=640, samples_per_image=3, transform=None, background_color=(255, 255, 255)):
        self.entity_manager = entity_manager
        self.image_size = image_size
        self.samples_per_image = samples_per_image
        self.transform = transform if transform else T.ToTensor()
        self.background_color = background_color

    def __len__(self):
        return len(self.entity_manager.entities)

    def __getitem__(self, idx):
        bg_np = np.full((self.image_size, self.image_size, 3), self.background_color, dtype=np.uint8)

        boxes = []
        labels = []

        for _ in range(self.samples_per_image):
            symbol = self.entity_manager.entities[idx].image
            symbol = Image.fromarray(symbol).convert("RGB").resize((32, 32))
            sx, sy = symbol.size
            x = random.randint(0, self.image_size - sx - 1)
            y = random.randint(0, self.image_size - sy - 1)
            bg_np[y:y+sy, x:x+sx] = np.array(symbol)
            boxes.append([x, y, x+sx, y+sy])
            labels.append(1)

        bg_tensor = self.transform(bg_np)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return bg_tensor, target