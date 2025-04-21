import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
from skimage.transform import resize
import cv2
from pathlib import Path

from ..core import Database
from ..utilities.observable import Observable


class SymbolDetector:
    def __init__(self, database: Database):
        self.database = database
        self.model = KNeighborsClassifier(n_neighbors=1)
        self.is_trained = False
        self.feature_vectors = []
        self.labels = []

    def _extract_features(self, image):
        """Simplified feature extraction for RGB or grayscale images"""
        # Resize to consistent size
        resized = resize(image, (64, 64))

        # Convert to uint8 if needed
        if resized.dtype == np.float64:
            resized = (resized * 255).astype(np.uint8)
        elif resized.dtype == np.float32:
            resized = (resized * 255).astype(np.uint8)

        # Convert to grayscale if needed
        if len(resized.shape) > 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

        # Extract HOG features
        features = hog(resized, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(1, 1), visualize=False)
        return features

    def _add_to_training_data(self, entity):
        features = self._extract_features(entity.image)
        self.feature_vectors.append(features)
        # Use entity manager name as label
        manager = next(mgr for mgr in self.database.get_entity_managers()
                       if entity in mgr.entities)
        self.labels.append(manager.entity_manager_name)
        self._retrain_model()

    def update_training_data(self):
        """Rebuild training data from scratch"""
        self.feature_vectors = []
        self.labels = []

        for manager in self.database.get_entity_managers():
            for entity in manager.entities:
                features = self._extract_features(entity.image)
                self.feature_vectors.append(features)
                self.labels.append(manager.entity_manager_name)

        self._retrain_model()

    def _retrain_model(self):
        if len(self.feature_vectors) > 0:
            self.model.fit(self.feature_vectors, self.labels)
            self.is_trained = True
        else:
            self.is_trained = False

    def detect_symbols(self, diagram_image):
        """Detect symbols in a user-provided diagram"""
        if not self.is_trained:
            raise ValueError("Model not trained - add entities to database first")

        # Preprocess the diagram
        gray = cv2.cvtColor(diagram_image, cv2.COLOR_BGR2GRAY) if len(diagram_image.shape) > 2 else diagram_image
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours (potential symbols)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_symbols = []

        for contour in contours:
            # Filter small contours (noise)
            if cv2.contourArea(contour) < 100:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Extract symbol
            symbol = diagram_image[y:y + h, x:x + w]

            # Extract features and predict
            features = self._extract_features(symbol)
            label = self.model.predict([features])[0]
            confidence = self.model.predict_proba([features]).max()

            detected_symbols.append({
                'symbol_type': label,
                'bounding_box': (x, y, w, h),
                'confidence': confidence,
                'image': symbol
            })

        return detected_symbols