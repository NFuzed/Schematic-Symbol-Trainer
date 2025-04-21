from ..utilities.observable import Observable
from .entity import Entity

class EntityManager:
    def __init__(self, entity_manager_name):
        self.entity_manager_name = entity_manager_name
        self.entities: [Entity] = []
        self.created_entity_observer = Observable()
        self.deleted_entity_observer = Observable()

    def create_entity(self, image):
        entity = Entity(image)
        self.entities.append(entity)
        self.created_entity_observer.notify(entity)
        return entity

    def remove_entity(self, entity : Entity):
        if entity in self.entities:
            result = self.entities.remove(entity)
            self.deleted_entity_observer.notify(entity)
            return result