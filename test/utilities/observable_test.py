from unittest import TestCase
from unittest.mock import Mock
from src.utilities.observable import Observable

class TestObserver(TestCase):
    def setUp(self):
        """Create an instance of Observer before each test."""
        self.observer = Observable()

    def test_notify_observers(self):
        """Test that notify_observers calls all added observers."""
        mock_observer_1 = Mock()
        mock_observer_2 = Mock()

        self.observer.bind(mock_observer_1)
        self.observer.bind(mock_observer_2)

        self.observer.notify("")

        mock_observer_1.assert_called_once()
        mock_observer_2.assert_called_once()