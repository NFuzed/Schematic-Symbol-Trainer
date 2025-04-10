import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
# todo: fix the imports

from src.utilities import Observable
from entity import Entity

class EntityManager:
    def __init__(self, entity_manager_name):
        self.entity_manager_name = entity_manager_name
        self.entities: [Entity] = []
        self.created_entity_observer = Observable()
        self.deleted_entity_observer = Observable()

    def create_entity(self, entity: Entity):
        self.created_entity_observer.value(entity)
        self.entities.append(entity)

    def remove_entity(self, entity : Entity):
        if entity in self.entities:
            self.deleted_entity_observer.value(entity)
            return self.entities.remove(entity)