from abc import ABC, abstractmethod

class Node(ABC):
    def __init__(self, name):
        self.name = name
        self.inputs = {}
        self.outputs = {}

    @abstractmethod
    def process(self):
        pass
