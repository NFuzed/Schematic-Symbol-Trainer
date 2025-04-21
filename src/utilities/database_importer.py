import pickle
from ..core.entity import Entity
from ..core.entity_manager import EntityManager

def import_database(database, import_path):
    if not database.clear_database():
        return

    with open(import_path, "rb") as f:
        saved_data = pickle.load(f)

    for group in saved_data:
        if group["type" == "ENTITY"]:
            manager = database.create_entity_manager(group["name"])
            for img in group["images"]:
                manager.create_entity(img)

        if group["type"] == "DIAGRAM":
            raise NotImplementedError

    print("Database imported successfully.")