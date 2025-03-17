from utilities import Observer
from entity import Entity

class EntityManager(Observer):
    def __init__(self):
        super().__init__()
        self.entities: dict[str, Entity] = {}

    def get_entity(self, id: str):
        return self.entities.get(id)

    def create_entity(self, id: str, entity: Entity):
        self.entities[id] = entity

    def remove_entity(self, id):
        return self.entities.pop(id)