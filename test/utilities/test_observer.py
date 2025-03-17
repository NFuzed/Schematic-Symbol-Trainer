from unittest import TestCase
from unittest.mock import Mock
from src.utilities.observer import Observer  # Assuming the class is in observer.py

class TestObserver(TestCase):
    def setUp(self):
        """Create an instance of Observer before each test."""
        self.observer = Observer()

    def test_add_observer(self):
        """Test that observers are added correctly."""
        mock_observer = Mock()

        self.observer.add_observer(mock_observer)
        self.assertIn(mock_observer, self.observer.observers)
        self.assertEqual(len(self.observer.observers), 1)

    def test_notify_observers(self):
        """Test that notify_observers calls all added observers."""
        mock_observer_1 = Mock()
        mock_observer_2 = Mock()

        self.observer.add_observer(mock_observer_1)
        self.observer.add_observer(mock_observer_2)

        self.observer.notify_observers()

        mock_observer_1.assert_called_once()
        mock_observer_2.assert_called_once()