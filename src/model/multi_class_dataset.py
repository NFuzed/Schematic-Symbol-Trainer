import os
import random
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import draw_bounding_boxes
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

from src.core.entity_manager import EntityManager

class MultiClassSymbolDataset(Dataset):
    def __init__(self, entity_managers: list[EntityManager], image_size=640, samples_per_image=3, rotation = False, flip = False, noise = False, background_color=(255, 255, 255)):
        self.entity_managers = sorted(entity_managers, key=lambda m: m.entity_manager_name)
        self.image_size = image_size
        self.samples_per_image = samples_per_image
        self.transform = T.ToTensor()
        self.background_color = background_color

        self.rotation = rotation
        self.flip = flip
        self.noise = noise

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
            if self.flip:
                if random.random() > 0.5:
                    symbol = symbol.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() > 0.5:
                    symbol = symbol.transpose(Image.FLIP_TOP_BOTTOM)

            if self.rotation:
                if random.random() > 0.5:
                    angle = random.randint(-20, 20)
                    symbol = symbol.rotate(angle, expand=True)

            if self.noise:
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