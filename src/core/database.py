import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from entity_manager import EntityManager
from src.utilities.observable import Observable

class Database:
    def __init__(self):
        self.entity_managers : [EntityManager] = []
        self.created_entity_manager_observer = Observable()
        self.destroyed_entity_manager_observer = Observable()

    def get_entity_managers(self):
        return self.entity_managers

    def create_entity_manager(self, entity_name):
        entity_manager = EntityManager(entity_name)
        self.entity_managers.append(entity_manager)
        self.created_entity_manager_observer.value(entity_manager)
        return entity_manager

    def delete_entity_manager(self, entity_manager : EntityManager):
        if entity_manager in self.entity_managers:
            self.destroyed_entity_manager_observer.value(entity_manager)
            self.entity_managers.remove(entity_manager)