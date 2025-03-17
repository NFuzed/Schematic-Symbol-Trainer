from typing import Callable, List

class Observer:
    def __init__(self):
        self.observers: List[Callable[[], None]] = []

    def notify_observers(self):
        for observer in self.observers:
            observer()

    def add_observer(self, observer: Callable[[], None]):
        self.observers.append(observer)