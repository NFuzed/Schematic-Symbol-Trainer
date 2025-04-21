from .diagram_manager import DiagramManager
from .entity_manager import EntityManager
from ..utilities.observable import Observable

class Database:
    def __init__(self):
        self.entity_managers : [EntityManager] = []
        self.diagrams = DiagramManager()
        self.created_entity_manager_observer = Observable()
        self.destroyed_entity_manager_observer = Observable()

    def get_entity_managers(self):
        return self.entity_managers

    def create_entity_manager(self, entity_name):
        entity_manager = EntityManager(entity_name)
        self.entity_managers.append(entity_manager)
        self.created_entity_manager_observer.notify(entity_manager)
        return entity_manager

    def delete_entity_manager(self, entity_manager : EntityManager):
        if entity_manager in self.entity_managers:
            self.destroyed_entity_manager_observer.notify(entity_manager)
            self.entity_managers.remove(entity_manager)

    def clear_database(self):
        for entity_manager in self.entity_managers.copy():
            self.delete_entity_manager(entity_manager)

        self.diagrams.clear_diagrams()
        return True