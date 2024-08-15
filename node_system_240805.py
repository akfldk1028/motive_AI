from abc import ABC, abstractmethod
from config import get_model_info
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import io
import base64
import os


class ModelStrategy(ABC):
    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def generate_image(self, pipe, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, batch_size):
        pass

class StableDiffusionStrategy(ModelStrategy):
    def load_model(self, model_path):
        return StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")

    def generate_image(self, pipe, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, batch_size):
        return pipe(
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size if negative_prompt else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
        ).images

class SDXLStrategy(ModelStrategy):
    def load_model(self, model_path):
        return StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")

    def generate_image(self, pipe, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, batch_size):
        return pipe(
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size if negative_prompt else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            original_size=(width, height),
            target_size=(width, height),
        ).images

class LoraStrategy(ModelStrategy):
    def load_model(self, model_path):
        base_model_info = get_model_info('SDXL')
        base_pipe = StableDiffusionXLPipeline.from_single_file(
            base_model_info['path'],
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        base_pipe.load_lora_weights(model_path)
        return base_pipe

    def generate_image(self, pipe, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, batch_size):
        return pipe(
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size if negative_prompt else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            original_size=(width, height),
            target_size=(width, height),
        ).images


class ModelFactory:
    @staticmethod
    def get_strategy(model_info):
        if model_info['type'] == 'checkpoint':
            if model_info['app'] == 'SDXL':
                return SDXLStrategy()
            else:
                return StableDiffusionStrategy()
        elif model_info['type'] == 'lora':
            return LoraStrategy()
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")

class Node(ABC):
    def __init__(self, name):
        self.name = name
        self.inputs = {}
        self.outputs = {}

    @abstractmethod
    def process(self):
        pass



class ModelLoaderNode(Node):
    def __init__(self, name, model_info):
        super().__init__(name)
        self.model_info = model_info
        self.inputs = {"model_path": model_info['path']}
        self.outputs = {"pipe": None}
        self.strategy = ModelFactory.get_strategy(model_info)

    def process(self):
        model_path = self.inputs["model_path"]
        print(f"모델 로딩 중: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        try:
            pipe = self.strategy.load_model(model_path)
            self.outputs["pipe"] = pipe
            print(f"모델 로딩 성공. 파이프라인 타입: {type(pipe)}")
        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {str(e)}")
            raise

class LoraNode(Node):
    def __init__(self, name, model_info):
        super().__init__(name)
        self.model_info = model_info
        self.inputs = {"pipe": None, "lora_path": model_info['path']}
        self.outputs = {"pipe": None}

    def process(self):
        pipe = self.inputs["pipe"]
        lora_path = self.inputs["lora_path"]
        pipe.load_lora_weights(lora_path)
        self.outputs["pipe"] = pipe


class TextPromptNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {"prompt": None, "negative_prompt": None}
        self.outputs = {"prompt": None, "negative_prompt": None}

    def process(self):
        self.outputs["prompt"] = self.inputs["prompt"]
        self.outputs["negative_prompt"] = self.inputs["negative_prompt"]


class ImageGeneratorNode(Node):
    def __init__(self, name, model_info):
        super().__init__(name)
        self.inputs = {
            "pipe": None, "prompt": None, "negative_prompt": None,
            "width": 512, "height": 512, "guidance_scale": 7.5,
            "num_inference_steps": 50, "batch_size": 1
        }
        self.outputs = {"images": None}
        self.strategy = ModelFactory.get_strategy(model_info)

    def process(self):
        print(f"{self.name}에서 이미지 생성 프로세스 시작")
        pipe = self.inputs["pipe"]

        if pipe is None:
            print(f"{self.name}에서 pipe가 None입니다")
            raise ValueError("pipe가 설정되지 않았습니다. 이전 노드와의 연결을 확인하세요.")

        print(f"pipe 객체 타입: {type(pipe)}")

        try:
            images = self.strategy.generate_image(
                pipe,
                self.inputs["prompt"],
                self.inputs["negative_prompt"],
                self.inputs["width"],
                self.inputs["height"],
                self.inputs["guidance_scale"],
                self.inputs["num_inference_steps"],
                self.inputs["batch_size"]
            )
            self.outputs["images"] = images
            print(f"{len(images)}개의 이미지 생성 성공")
        except Exception as e:
            print(f"이미지 생성 중 오류 발생: {str(e)}")
            print(f"파이프라인 속성: {dir(pipe)}")
            raise

class ImageEncoderNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {"images": None}
        self.outputs = {"encoded_images": None}

    def process(self):
        images = self.inputs["images"]
        encoded_images = []
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            encoded_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
        self.outputs["encoded_images"] = encoded_images

# 새로운 노드 유형 예시
class ControlNetNode(Node):
    def __init__(self, name, control_type):
        super().__init__(name)
        self.control_type = control_type
        self.inputs = {"pipe": None, "control_image": None}
        self.outputs = {"pipe": None}

    def process(self):
        # ControlNet 로직 구현
        pass

class ImageToImageNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {"pipe": None, "init_image": None, "strength": 0.8}
        self.outputs = {"pipe": None}

    def process(self):
        # Image-to-Image 로직 구현
        pass


class Graph:
    def __init__(self):
        self.nodes = []
        self.connections = []

    def add_node(self, node):
        self.nodes.append(node)

    def connect(self, output_node, output_key, input_node, input_key):
        self.connections.append((output_node, output_key, input_node, input_key))
        print(f"{output_node.name}.{output_key}를 {input_node.name}.{input_key}에 연결")

    def process(self):
        print("그래프 처리 시작")
        print(self.nodes)
        for i, node in enumerate(self.nodes):
            print(f"노드 {i} 처리 중: {node.name}")
            node.process()

            for out_node, out_key, in_node, in_key in self.connections:
                if out_node == node:
                    in_node.inputs[in_key] = out_node.outputs[out_key]

        print("그래프 처리 완료")


def create_graph_for_model(model_name, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, batch_size):
    graph = Graph()
    model_info = get_model_info(model_name)
    print(f"model_info: {model_info}")
    # model_info: {'type': 'checkpoint', 'path': './models/checkpoints/architecturerealmix_v11.safetensors',
    #              'app': 'CheckPoint'}

    if not model_info:
        raise ValueError(f"알 수 없는 모델: {model_name}")

    model_loader = ModelLoaderNode("model_loader", model_info)
    graph.add_node(model_loader)

    if model_info['type'] == 'lora':
        lora_node = LoraNode("lora_loader", model_info)
        graph.add_node(lora_node)
        graph.connect(model_loader, "pipe", lora_node, "pipe")
        last_model_node = lora_node
    else:
        last_model_node = model_loader

    text_prompt = TextPromptNode("text_prompt")
    image_generator = ImageGeneratorNode("image_generator", model_info)
    image_encoder = ImageEncoderNode("image_encoder")

    graph.add_node(text_prompt)
    graph.add_node(image_generator)
    graph.add_node(image_encoder)

    graph.connect(last_model_node, "pipe", image_generator, "pipe")
    graph.connect(text_prompt, "prompt", image_generator, "prompt")
    graph.connect(text_prompt, "negative_prompt", image_generator, "negative_prompt")
    graph.connect(image_generator, "images", image_encoder, "images")

    # 입력값 설정
    text_prompt.inputs["prompt"] = prompt
    text_prompt.inputs["negative_prompt"] = negative_prompt
    image_generator.inputs["width"] = width
    image_generator.inputs["height"] = height
    image_generator.inputs["guidance_scale"] = guidance_scale
    image_generator.inputs["num_inference_steps"] = num_inference_steps
    image_generator.inputs["batch_size"] = batch_size

    return graph

