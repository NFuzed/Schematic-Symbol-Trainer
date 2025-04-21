import pickle
import os


def export_database(database, export_path):
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    # Collect entity manager data
    data = []
    for manager in database.get_entity_managers():
        entities = [entity.image for entity in manager.entities]
        data.append({
            "name": manager.entity_manager_name,
            "images": entities,
            "type": "ENTITY"
        })

    for diagram in database.diagrams.diagrams_file_paths:
        data.append({
            "diagram": diagram,
            "type": "DIAGRAM"
        })

    # Write to the given path
    with open(export_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Database exported to {export_path}")