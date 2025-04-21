from unittest import TestCase
from unittest.mock import Mock
from src.core.entity_manager import EntityManager

class TestEntityManager(TestCase):
    def setUp(self):
        """Create a mock entity manager instance"""
        self.entity_manager = EntityManager("Test Manager")

    def test_modifying_entities(self):
        """Test that entities are correctly added and removed"""
        self.assertEqual(len(self.entity_manager.entities), 0)

        entity_1 = self.entity_manager.create_entity(None)
        entity_2 = self.entity_manager.create_entity(None)
        entity_3 = self.entity_manager.create_entity(None)

        self.assertEqual(len(self.entity_manager.entities), 3)
        self.assertListEqual(self.entity_manager.entities, [entity_1, entity_2, entity_3])

        self.entity_manager.remove_entity(entity_1)
        self.entity_manager.remove_entity(entity_2)
        self.entity_manager.remove_entity(entity_3)

        self.assertEqual(0, len(self.entity_manager.entities))

    def test_subscribers_on_updates(self):
        """Test that subscribers are notified when entities are added and removed"""

        entity_creation_subscriber = Mock()
        entity_deleted_subscriber = Mock()

        self.entity_manager.created_entity_observer.bind(entity_creation_subscriber)
        created_entity = self.entity_manager.create_entity(None)
        entity_creation_subscriber.assert_called_once_with(created_entity)

        self.entity_manager.deleted_entity_observer.bind(entity_deleted_subscriber)
        self.entity_manager.remove_entity(created_entity)
        entity_deleted_subscriber.assert_called_once_with(created_entity)