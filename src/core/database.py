from core.entity_manager import EntityManager


class Database:
    def __init__(self):
        self.entity_manager = EntityManager()

    def get_entity_manager(self):
        return self.entity_manager