from ..utilities.observable import Observable
from .entity import Entity

class DiagramManager:
    def __init__(self):
        self.diagrams_file_paths = []
        self.added_diagram_observer = Observable()
        self.deleted_diagram_observer = Observable()

    def add_diagram(self, diagram_file_path):
        self.diagrams_file_paths.append(diagram_file_path)
        self.added_diagram_observer.notify(diagram_file_path)
        return diagram_file_path

    def remove_diagram(self, diagram_file_path):
        if diagram_file_path in self.diagrams_file_paths:
            result = self.diagrams_file_paths.remove(diagram_file_path)
            self.deleted_diagram_observer.notify(diagram_file_path)
            return result

    def clear_diagrams(self):
        for diagram_file_path in self.diagrams_file_paths.copy():
            self.remove_diagram(diagram_file_path)