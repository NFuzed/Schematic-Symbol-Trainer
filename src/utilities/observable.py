
class Observable:
    def __init__(self):
        self._value = None
        self._callbacks = []

    def notify(self, new_value):
        self._value = new_value
        for callback in self._callbacks:
            callback(new_value)

    def bind(self, callback):
        self._callbacks.append(callback)