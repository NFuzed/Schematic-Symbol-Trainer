from unittest import TestCase
from src.utilities import database_exporter, database_importer
from src.core import Database, EntityManager, Entity
from numpy import ndarray
import numpy as np
import os
import tempfile

class TestDatabaseExporterAndImporter(TestCase):
    def setUp(self):
        self.database: Database = Database()
        self.rng = np.random.default_rng(seed=42)

        for i in range(2):
            manager = self.database.create_entity_manager(f"Symbol_{i}")
            for _ in range(3):  # 3 images per manager
                img = self.rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
                manager.create_entity(img)

    def test_export_and_import_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "test_db.pkl")
            database_exporter.export_database(self.database, export_path)

            imported_db = Database()
            database_importer.import_database(imported_db, export_path)

            original = self.database.get_entity_managers()
            restored = imported_db.get_entity_managers()

            self.assertEqual(len(original), len(restored), "EntityManager count mismatch")
            for o_mgr, r_mgr in zip(original, restored):
                self.assertEqual(o_mgr.entity_manager_name, r_mgr.entity_manager_name)
                self.assertEqual(len(o_mgr.entities), len(r_mgr.entities), "Entity count mismatch")

                for o_ent, r_ent in zip(o_mgr.entities, r_mgr.entities):
                    self.assertIsInstance(r_ent.image, ndarray)
                    self.assertTrue(np.array_equal(o_ent.image, r_ent.image), "Image data mismatch")
