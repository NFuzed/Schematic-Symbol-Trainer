import cv2
import numpy as np
from src.core.database import Database

class EntityDetector:
    def __init__(self, database: Database, threshold=0.8, min_votes=2):
        self.database = database
        self.threshold = threshold
        self.min_votes = min_votes  # Minimum number of matches required to confirm detection

    def detect_entities(self, diagram_image: np.ndarray):
        gray_diagram = cv2.cvtColor(diagram_image, cv2.COLOR_BGR2GRAY)
        detection_votes = []

        for entity_manager in self.database.get_entity_managers():
            entity_type = entity_manager.entity_manager_name
            entity_matches = []

            for entity in entity_manager.entities:
                gray_template = cv2.cvtColor(entity.image, cv2.COLOR_BGR2GRAY)
                w, h = gray_template.shape[::-1]

                result = cv2.matchTemplate(gray_diagram, gray_template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(result >= self.threshold)

                for pt in zip(*loc[::-1]):
                    entity_matches.append((pt, (w, h)))

            # Aggregate matches to find overlapping detections
            confirmed_detections = self._aggregate_matches(entity_matches)

            for det in confirmed_detections:
                detection_votes.append({
                    'entity_type': entity_type,
                    'position': det['position'],
                    'size': det['size'],
                    'votes': det['votes']
                })

        # Filter by minimum votes (noise filtering)
        filtered_detections = [d for d in detection_votes if d['votes'] >= self.min_votes]

        return filtered_detections

    def _aggregate_matches(self, matches):
        aggregated = []

        for pos, size in matches:
            found = False
            for existing in aggregated:
                if self._positions_close(pos, existing['position']):
                    existing['votes'] += 1
                    found = True
                    break
            if not found:
                aggregated.append({'position': pos, 'size': size, 'votes': 1})

        return aggregated

    @staticmethod
    def _positions_close(pos1, pos2, tolerance=10):
        return np.linalg.norm(np.array(pos1) - np.array(pos2)) <= tolerance

