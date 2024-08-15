from .base_node import Node

class TextPromptNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {"prompt": None, "negative_prompt": None}
        self.outputs = {"prompt": None, "negative_prompt": None}

    def process(self):
        self.outputs["prompt"] = self.inputs["prompt"]
        self.outputs["negative_prompt"] = self.inputs["negative_prompt"]