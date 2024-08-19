from abc import ABC, abstractmethod
from config import get_model_info, AI_MODELS
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, \
    UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler, DPMSolverSDEScheduler, ControlNetModel, \
    StableDiffusionControlNetPipeline, UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline
import cv2
from transformers import pipeline
from palette import palette

import io
import base64
import os
from nodes.base_node import Node
from nodes.text_prompt_node import TextPromptNode
from transformers import CLIPTextModel, CLIPTokenizer
import safetensors
from safetensors.torch import load_file
import traceback
import diffusers
from diffusers import DiffusionPipeline
from peft import PeftModel, LoraConfig
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from torchvision.transforms import Resize, ToTensor, Normalize
# https://github.com/LeandroBerlin/SDXL-base-with-refiner/blob/main/SDXL_base_with_refiner.ipynb

# @title
from diffusers import DiffusionPipeline

print(f"Diffusers version: {diffusers.__version__}")


class ModelStrategy(ABC):
    @abstractmethod
    def load_model(self, model_path):
        pass

    @abstractmethod
    def generate_image(self, pipe, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps,
                       batch_size):
        pass


# class StableDiffusionStrategy(ModelStrategy):
#     def load_model(self, model_path):
#         return StableDiffusionPipeline.from_single_file(
#             model_path,
#             torch_dtype=torch.float16,
#             use_safetensors=True
#         ).to("cuda")
#
#     def generate_image(self, pipe, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps,
#                        batch_size):
#         return pipe(
#             prompt=[prompt] * batch_size,
#             negative_prompt=[negative_prompt] * batch_size if negative_prompt else None,
#             width=width,
#             height=height,
#             guidance_scale=guidance_scale,
#             num_inference_steps=num_inference_steps,
#         ).images


class LoraControlNetStrategy(ModelStrategy):
    def __init__(self, model_info, controlnet_info):
        self.model_info = model_info
        self.controlnet_info = controlnet_info

    def load_model(self, model_path):
        base_checkpoint = self.model_info['baseCheckpoint']
        base_model_info = get_model_info(base_checkpoint)

        controlnets = []
        for control_info in self.controlnet_info:
            controlnet = ControlNetModel.from_pretrained(
                control_info['model_path'],
                torch_dtype=torch.float16
            ).to("cuda")
            controlnets.append(controlnet)

        if base_model_info['app'] == 'SDXL':
            base_pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                base_model_info['path'],
                controlnet=controlnets,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to("cuda")
        else:
            base_pipe = StableDiffusionControlNetPipeline.from_single_file(
                base_model_info['path'],
                controlnet=controlnets,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None
            ).to("cuda")

        base_pipe.load_lora_weights(model_path)
        base_pipe.scheduler = UniPCMultistepScheduler.from_config(base_pipe.scheduler.config)
        # base_pipe.enable_model_cpu_offload()

        return base_pipe

    def generate_image(self, pipe, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps,
                       batch_size, **kwargs):
        control_images = kwargs.get("image", [])
        if not isinstance(control_images, list):
            control_images = [control_images]

        return pipe(
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size if negative_prompt else None,
            image=control_images,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
        ).images


class StableDiffusionStrategy(ModelStrategy):
    def load_model(self, model_path):
        return StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")

    def generate_image(self, pipe, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps,
                       batch_size, **kwargs):
        generate_kwargs = {
            "prompt": [prompt] * batch_size,
            "negative_prompt": [negative_prompt] * batch_size if negative_prompt else None,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        }

        if "image" in kwargs and isinstance(pipe, StableDiffusionControlNetPipeline):
            generate_kwargs["image"] = kwargs["image"]
            print(f"Applying ControlNet with image of size: {kwargs['image'].size}")

        return pipe(**generate_kwargs).images


class SDXLControlNetStrategy(ModelStrategy):
    def __init__(self, controlnet_info):
        self.base_model = None
        self.refiner_model = None
        self.controlnet_info = controlnet_info
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path):
        print(f"SDXL ControlNet 모델 로딩 중: {model_path}")
        try:
            # ControlNet 모델 로드
            controlnets = []
            for control_info in self.controlnet_info:
                controlnet = ControlNetModel.from_pretrained(
                    control_info['model_path'],
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
                controlnets.append(controlnet)
            vae = AutoencoderKL.from_pretrained("./motive_v1/models/madebyollin/sdxl-vae-fp16-fix",
                                                torch_dtype=torch.float16)
            # self.refiner_model.enable_model_cpu_offload()

            # Base 모델 로드
            self.base_model = StableDiffusionXLControlNetPipeline.from_pretrained(
                model_path,
                controlnet=controlnets,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                vae=vae,
            )
            # self.base_model.scheduler = UniPCMultistepScheduler.from_config(self.base_model.scheduler.config)
            self.base_model.scheduler = EulerAncestralDiscreteScheduler.from_config(self.base_model.scheduler.config)
            self.base_model.enable_model_cpu_offload()

            # Refiner 모델 로드
            refiner_path = model_path.replace("SDXL_base_model", "SDXL_refiner_model")
            print(f"SDXL Refiner 모델 로딩 중: {refiner_path}")
            self.refiner_model = DiffusionPipeline.from_pretrained(
                refiner_path,
                text_encoder_2=self.base_model.text_encoder_2,
                vae=self.base_model.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            self.refiner_model.scheduler = EulerDiscreteScheduler.from_config(self.refiner_model.scheduler.config)
            self.refiner_model.enable_model_cpu_offload()

            print("SDXL ControlNet Base 및 Refiner 모델 로딩 성공")
        except Exception as e:
            print(f"SDXL ControlNet 모델 로딩 중 오류 발생: {str(e)}")
            print(f"오류 타입: {type(e).__name__}")
            print(f"오류 발생 위치:\n{traceback.format_exc()}")
            raise

        return self.base_model

    def generate_image(self, pipe, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps,
                       batch_size, **kwargs):

        if self.base_model is None or self.refiner_model is None:
            raise ValueError("SDXL 모델이 로드되지 않았습니다. load_model을 먼저 호출하세요.")

        width, height = self.get_optimal_resolution(width, height)
        print(f"선택된 해상도: {width}x{height}")

        # high_noise_frac = 0.8
        # total_steps = num_inference_steps
        total_steps = 35

        # end_at_step = 20
        # high_noise_frac = end_at_step / total_steps

        base_steps = kwargs.get('base_steps', 35)
        refiner_steps = kwargs.get('refiner_steps', 25)
        end_at_step = kwargs.get('end_at_step', 25)
        controlnet_conditioning_scale = 0.5  # recommended for good generalization

        control_images = kwargs.get("image", [])
        if not isinstance(control_images, list):
            control_images = [control_images]

        # Base 모델로 이미지 생성
        latents = self.base_model(
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size,
            image=control_images,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=total_steps,
            output_type="latent",
            denoising_end=end_at_step / total_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images

        # Refiner 모델로 이미지 개선
        images = self.refiner_model(
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size,
            image=latents,
            num_inference_steps=total_steps,
            guidance_scale=guidance_scale,
            denoising_start=end_at_step / total_steps,
        ).images

        return images

    def get_optimal_resolution(self, width, height):
        resolutions = [
            (1024, 1024), (1152, 896), (1216, 832), (1344, 768), (1536, 640),
            (640, 1536), (768, 1344), (832, 1216), (896, 1152)
        ]
        return min(resolutions, key=lambda res: abs(res[0] - width) + abs(res[1] - height))


class SDXLStrategy(ModelStrategy):
    def __init__(self):
        self.base_model = None
        self.refiner_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path):
        print(f"SDXL Base 모델 로딩 중: {model_path}")
        try:
            self.base_model = DiffusionPipeline.from_pretrained(
                model_path, torch_dtype=torch.float16, variant="fp16",
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
            self.base_model.scheduler = EulerAncestralDiscreteScheduler.from_config(self.base_model.scheduler.config)
            self.base_model.to("cuda")
            # self.base_model.enable_model_cpu_offload()
            # self.base_model.enable_vae_slicing()

            refiner_path = model_path.replace("SDXL_base_model", "SDXL_refiner_model")
            print(f"SDXL Refiner 모델 로딩 중: {refiner_path}")
            self.refiner_model = DiffusionPipeline.from_pretrained(
                refiner_path,
                text_encoder_2=self.base_model.text_encoder_2,
                vae=self.base_model.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                low_cpu_mem_usage=True,

            )
            self.refiner_model.scheduler = EulerDiscreteScheduler.from_config(self.refiner_model.scheduler.config)
            self.refiner_model.to("cuda")
            # self.refiner_model.enable_model_cpu_offload()
            # self.refiner_model.enable_vae_slicing()

            print("SDXL Base 및 Refiner 모델 로딩 성공")
        except Exception as e:
            print(f"SDXL 모델 로딩 중 오류 발생: {str(e)}")
            print(f"오류 타입: {type(e).__name__}")
            print(f"오류 발생 위치:\n{traceback.format_exc()}")
            raise

        return self.base_model

    def generate_image(self, pipe, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps,
                       batch_size, seed=None, control_after_generate="fixed"):
        if self.base_model is None or self.refiner_model is None:
            raise ValueError("SDXL 모델이 로드되지 않았습니다. load_model을 먼저 호출하세요.")

        width, height = self.get_optimal_resolution(width, height)
        print(f"선택된 해상도: {width}x{height}")

        # high_noise_frac = 0.8
        total_steps = num_inference_steps
        end_at_step = 20
        print(batch_size)
        high_noise_frac = end_at_step / total_steps

        # 시드 처리
        if seed is None:
            seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()

        generators = []
        for i in range(batch_size):
            if control_after_generate == "fixed":
                generators.append(torch.Generator(device=self.device).manual_seed(seed))
            elif control_after_generate == "increment":
                generators.append(torch.Generator(device=self.device).manual_seed(seed + i))
            elif control_after_generate == "randomize":  # "randomize"
                generators.append(
                    torch.Generator(device=self.device).manual_seed(torch.randint(0, 2 ** 32 - 1, (1,)).item()))
            # generator=generators,

        try:
            # Base 모델
            latents = self.base_model(
                prompt=[prompt] * batch_size,
                negative_prompt=[negative_prompt] * batch_size,
                width=width,
                height=height,
                num_inference_steps=total_steps,
                guidance_scale=guidance_scale,
                output_type="latent",
                denoising_end=high_noise_frac,
            ).images

            # Refiner 모델
            images = self.refiner_model(
                prompt=[prompt] * batch_size,
                negative_prompt=[negative_prompt] * batch_size,
                image=latents,
                num_inference_steps=total_steps,
                guidance_scale=guidance_scale,
                denoising_start=high_noise_frac,
                strength=1 - (end_at_step / total_steps),  # strength를 사용하여 시작 지점 조절
            ).images

            return images
        except Exception as e:
            print(f"이미지 생성 중 오류 발생: {str(e)}")
            print(f"오류 타입: {type(e).__name__}")
            print(f"오류 발생 위치:\n{traceback.format_exc()}")
            raise

    def get_optimal_resolution(self, width, height):
        resolutions = [
            (1024, 1024), (1152, 896), (1216, 832), (1344, 768), (1536, 640),
            (640, 1536), (768, 1344), (832, 1216), (896, 1152)
        ]
        return min(resolutions, key=lambda res: abs(res[0] - width) + abs(res[1] - height))


class RealisticVisionStrategy(ModelStrategy):
    def load_model(self, model_path):
        pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)
        return pipe

    def generate_image(self, pipe, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps,
                       batch_size):
        return pipe(
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size if negative_prompt else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images


class LoraStrategy(ModelStrategy):
    def __init__(self, model_info):
        self.model_info = model_info

    def load_model(self, model_path):
        base_checkpoint = self.model_info['baseCheckpoint']
        print(f"Base checkpoint: {base_checkpoint}")
        base_model_info = get_model_info(base_checkpoint)
        print(f"Base model info: {base_model_info}")

        if base_model_info['app'] == 'SDXL':
            base_pipe = StableDiffusionXLPipeline.from_single_file(
                base_model_info['path'],
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to("cuda")
        else:
            base_pipe = StableDiffusionPipeline.from_single_file(
                base_model_info['path'],
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None  # 안전 검사기를 비활성화합니다.
            ).to("cuda")

        # Load and apply LoRA weights
        print(f"Loading LoRA weights from: {model_path}")
        base_pipe.load_lora_weights(model_path)

        # Set up the scheduler
        base_pipe.scheduler = DPMSolverSDEScheduler.from_config(
            base_pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++"
        )

        return base_pipe

    def generate_image(self, pipe, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps,
                       batch_size):
        return pipe(
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size if negative_prompt else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
        ).images


class ControlNetStrategy(ModelStrategy):
    def __init__(self, model_info, controlnet_info):
        self.model_info = model_info
        self.controlnet_info = controlnet_info

    def load_model(self, model_path):
        controlnets = []
        for control_info in self.controlnet_info:
            controlnet = ControlNetModel.from_pretrained(
                control_info['model_path'],
                torch_dtype=torch.float16
            ).to("cuda")
            controlnets.append(controlnet)

        pipe = StableDiffusionControlNetPipeline.from_single_file(
            model_path,
            controlnet=controlnets,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_model_cpu_offload()

        return pipe

    def generate_image(self, pipe, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps,
                       batch_size, **kwargs):
        control_images = kwargs.get("image", [])
        print("Control images:", control_images)  # 디버그용 출력
        if not isinstance(control_images, list):
            control_images = [control_images]

        if len(control_images) == 0:
            raise ValueError("No control images provided for ControlNet")

        return pipe(
            prompt=[prompt] * batch_size,
            negative_prompt=[negative_prompt] * batch_size if negative_prompt else None,
            image=control_images,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
        ).images


class ModelFactory:
    @staticmethod
    def get_strategy(model_info, controlnet_info=None):
        print(controlnet_info)
        print("-------------------")
        if model_info['type'] == 'checkpoint':
            if model_info['app'] == 'SDXL':
                if controlnet_info:
                    return SDXLControlNetStrategy(controlnet_info)
                else:
                    return SDXLStrategy()
            elif (model_info['app'] == 'RealisticVision') or (model_info['app'] == 'ReRender'):
                if controlnet_info:
                    return ControlNetStrategy(model_info, controlnet_info)
                else:
                    return RealisticVisionStrategy()
            else:
                if controlnet_info:
                    return ControlNetStrategy(model_info, controlnet_info)
                else:
                    return StableDiffusionStrategy()

        elif model_info['type'] == 'lora':
            if controlnet_info:
                return LoraControlNetStrategy(model_info, controlnet_info)
            else:
                return LoraStrategy(model_info)  # 다른 LoRA 모델들도 동일한 전략 사용
        else:
            raise ValueError(f"Unknown model type: {model_info['type']}")


class ModelLoaderNode(Node):
    def __init__(self, name, model_info, controlnet_info=None):
        super().__init__(name)
        self.model_info = model_info
        self.controlnet_info = controlnet_info
        self.inputs = {"model_path": model_info['path']}
        self.outputs = {"pipe": None, "strategy": None}
        print(f"ModelLoaderNode initialized with controlnet_info: {controlnet_info}")  # 디버깅을 위해 추가

    def process(self):
        model_path = self.inputs["model_path"]
        print(f"모델 로딩 중: {model_path}")
        print(f"ControlNet 정보: {self.controlnet_info}")  # 디버깅을 위해 추가

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        try:
            strategy = ModelFactory.get_strategy(self.model_info, self.controlnet_info)
            pipe = strategy.load_model(model_path)
            self.outputs["pipe"] = pipe
            self.outputs["strategy"] = strategy
            print(f"모델 로딩 성공. 파이프라인 타입: {type(pipe)}")
        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {str(e)}")
            raise


class ImageGeneratorNode(Node):
    def __init__(self, name, model_info):
        super().__init__(name)
        self.inputs = {
            "pipe": None, "strategy": None, "prompt": None, "negative_prompt": None,
            "width": 512, "height": 512, "guidance_scale": 7.5,
            "num_inference_steps": 50, "batch_size": 1, "control_images": None
        }
        self.outputs = {"images": None}

    def process(self):
        print(f"{self.name}에서 이미지 생성 프로세스 시작")
        pipe = self.inputs["pipe"]
        strategy = self.inputs["strategy"]
        control_images = self.inputs.get("control_images")

        if pipe is None or strategy is None:
            raise ValueError("pipe 또는 strategy가 설정되지 않았습니다. 이전 노드와의 연결을 확인하세요.")

        try:
            generate_kwargs = {
                "prompt": self.inputs["prompt"],
                "negative_prompt": self.inputs["negative_prompt"],
                "width": self.inputs["width"],
                "height": self.inputs["height"],
                "guidance_scale": self.inputs["guidance_scale"],
                "num_inference_steps": self.inputs["num_inference_steps"],
                "batch_size": self.inputs["batch_size"],
            }

            print(f"Control images input: {control_images}")

            if control_images is not None:
                if isinstance(control_images, Image.Image):
                    generate_kwargs["image"] = control_images
                    print(f"ControlNet 이미지가 적용됩니다.")
                elif isinstance(control_images, list):
                    generate_kwargs["image"] = control_images
                    print(f"ControlNet 이미지가 적용됩니다. 이미지 개수: {len(control_images)}")
                else:
                    print(f"Warning: Unexpected control_images type: {type(control_images)}")
                    raise ValueError("Unexpected control_images type")
            else:
                print("No ControlNet images provided")

            print("Generate kwargs:", generate_kwargs)
            images = strategy.generate_image(pipe, **generate_kwargs)

            if not images:
                raise ValueError("이미지 생성에 실패했습니다.")

            self.outputs["images"] = images
            print(f"{len(images)}개의 이미지 생성 성공")
        except Exception as e:
            print(f"이미지 생성 중 오류 발생: {str(e)}")
            raise


class ImageEncoderNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {"images": None}
        self.outputs = {"encoded_images": None}

    def process(self):
        images = self.inputs["images"]
        print(f"ImageEncoderNode 입력 이미지: {type(images)}")

        if images is None:
            print("경고: 입력 이미지가 None입니다.")
            self.outputs["encoded_images"] = None
            return

        if not isinstance(images, list):
            images = [images]

        encoded_images = []
        for i, image in enumerate(images):
            if image is None:
                print(f"경고: 이미지 {i}가 None입니다.")
                continue

            try:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                encoded_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
            except Exception as e:
                print(f"이미지 {i} 인코딩 중 오류 발생: {str(e)}")

        self.outputs["encoded_images"] = encoded_images
        print(f"인코딩된 이미지 수: {len(encoded_images)}")


class ImageEncoderNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {"images": None}
        self.outputs = {"encoded_images": None}

    def process(self):
        images = self.inputs["images"]
        print(f"ImageEncoderNode 입력 이미지: {type(images)}")

        if images is None:
            print("경고: 입력 이미지가 None입니다.")
            self.outputs["encoded_images"] = None
            return

        if not isinstance(images, list):
            images = [images]

        encoded_images = []
        for i, image in enumerate(images):
            if image is None:
                print(f"경고: 이미지 {i}가 None입니다.")
                continue

            try:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                encoded_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
            except Exception as e:
                print(f"이미지 {i} 인코딩 중 오류 발생: {str(e)}")

        self.outputs["encoded_images"] = encoded_images
        print(f"인코딩된 이미지 수: {len(encoded_images)}")


class ControlNetPreprocessorNode(Node):
    def __init__(self, name, preprocessor_type, imgtoimg, width, height):
        super().__init__(name)
        self.preprocessor_type = preprocessor_type
        self.inputs = {"image": imgtoimg}
        self.outputs = {"processed_image": None}
        self.depth_estimator = None
        self.width = width
        self.height = height
    def process(self):
        print(f"ControlNetPreprocessorNode processing with type: {self.preprocessor_type}")
        if self.inputs["image"] is None:
            raise ValueError("No input image provided to ControlNetPreprocessorNode")

        print(self.preprocessor_type)
        print("----------------------------------------")
        if self.preprocessor_type == "canny":
            self.outputs["processed_image"] = self.canny_preprocess()
        elif self.preprocessor_type == "depth":
            self.outputs["processed_image"] = self.depth_preprocess()
        elif self.preprocessor_type == "SDXLdepth":
            self.outputs["processed_image"] = self.sdxl_depth_preprocess()
        elif self.preprocessor_type == "segmentation":
            self.outputs["processed_image"] = self.segmentation_preprocess()
        else:
            raise ValueError(f"Unsupported preprocessor type: {self.preprocessor_type}")

        if self.outputs["processed_image"] is None:
            raise ValueError("Preprocessor failed to produce an output image")

        print(f"ControlNetPreprocessorNode successfully processed image")
        print(f"ControlNetPreprocessorNode '{self.name}' processed image:")
        print(f"  Type: {self.preprocessor_type}")
        print(f"  Input image: {self.inputs['image']}")
        print(f"  Processed image: {self.outputs['processed_image']}")

    def sdxl_depth_preprocess(self):
        from transformers import DPTFeatureExtractor, DPTForDepthEstimation
        print("0000000000000000000000000000")

        depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
        print("왜안됨? 시발?")
        print("0000000000000000000000000000")
        image = self.inputs["image"]

        # 이미지가 PIL Image가 아니면 변환
        # if not isinstance(image, Image.Image):
        #     image = Image.fromarray(image.astype('uint8'), 'RGB')
        # image = image.resize((self.width, self.height))

        print(f"Input image type: {type(image)}")

        # 이미지 타입 체크 및 변환
        # PNG 파일 처리
        if isinstance(image, Image.Image):
            print(f"Input image mode: {image.mode}")
            if image.mode == 'RGBA':
                print("Converting RGBA to RGB")
                image = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.shape[-1] == 4:  # RGBA
                print("Converting RGBA array to RGB")
                image = image[..., :3]
            image = Image.fromarray(image)

        image = image.resize((self.width, self.height))
        print(f"Resized image size: {image.size}")


        print("Extracting features")
        try:
            image_features = feature_extractor(images=image, return_tensors="pt")
            print(f"Feature extraction successful. Shape: {image_features.pixel_values.shape}")
        except Exception as e:
            print(f"Error during feature extraction: {str(e)}")
            raise

        print("Moving image to CUDA")
        image = image_features.pixel_values.to("cuda")

        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(self.height, self.width),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    def softedge_preprocess(self):
        from controlnet_aux import PidiNetDetector, HEDdetector

        processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
        processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
        image = np.array(self.inputs["image"])
        control_image = processor(image, safe=True)
        control_image.save("./example/control.png")
        return control_image

    def lineart_preprocess(self):
        from controlnet_aux import LineartDetector

        image = np.array(self.inputs["image"])
        image = image.resize((512, 512))
        processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        control_image = processor(image)
        return control_image

    def segmentation_preprocess(self):
        image_processor = AutoImageProcessor.from_pretrained("./motive_v1/models/openmmlab/upernet-convnext-small")
        image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            "./motive_v1/models/openmmlab/upernet-convnext-small")
        image = np.array(self.inputs["image"])

        pixel_values = image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = image_segmentor(pixel_values)

        # NumPy 배열의 shape를 사용하여 target_sizes를 지정합니다.
        # height와 width의 순서를 바꿔야 할 수 있습니다. 필요에 따라 조정하세요.
        target_size = image.shape[:2][::-1]  # (width, height)
        seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[target_size])[0]

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        control_image = Image.fromarray(color_seg)

        output_folder = './example'  # 저장할 폴더 경로
        output_filename = 'saved_seg_image.png'  # 저장할 파일 이름
        output_path = os.path.join(output_folder, output_filename)
        control_image.save(output_path)
        return control_image

    def canny_preprocess(self):
        image = np.array(self.inputs["image"])
        image = cv2.resize(image, (self.width, self.height))

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)

        # output_folder = './example'  # 저장할 폴더 경로
        # output_filename = 'saved_canny_image.png'  # 저장할 파일 이름
        # output_path = os.path.join(output_folder, output_filename)
        # control_image.save(output_path)
        return control_image

    def depth_preprocess(self):
        image = self.inputs["image"]
        image = image.resize((self.width, self.height))

        # GPU가 있으면 'cuda'를 사용하고, 그렇지 않으면 'cpu'를 사용합니다.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 모델을 로드할 때 'device'를 명시적으로 설정합니다.
        depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-large", device=device)
        # 깊이 추정을 수행합니다.
        image = depth_estimator(image)['depth']
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)
        # 이미지 저장 경로 설정
        output_folder = './example'  # 저장할 폴더 경로
        output_filename = 'saved_depth_image.png'  # 저장할 파일 이름

        # 이미지 저장 경로 설정
        output_path = os.path.join(output_folder, output_filename)

        # 이미지 저장
        control_image.save(output_path)

        return control_image


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
            print(f"노드 {i} 처리 완료: {node.name}")

            for out_node, out_key, in_node, in_key in self.connections:
                if out_node == node:
                    in_node.inputs[in_key] = out_node.outputs[out_key]
                    print(f"연결: {out_node.name}.{out_key} -> {in_node.name}.{in_key}")

        print("그래프 처리 완료")
        print(f"마지막 노드: {self.nodes[-1].name}")
        # print(f"마지막 노드 출력: {self.nodes[-1].outputs}")


class LoadImageNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {"image_input": None}
        self.outputs = {"image": None}

    def process(self):
        from PIL import Image
        import numpy as np

        image_input = self.inputs["image_input"]
        print(f"LoadImageNode received input of type: {type(image_input)}")

        if image_input is None:
            raise ValueError("No image input provided to LoadImageNode")

        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input.astype('uint8'), 'RGB')
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            image_np = np.array(image)
            self.outputs["image"] = image_np

            # self.outputs["image"] = image
            print(f"LoadImageNode successfully loaded image of size: {image.size}")
        except Exception as e:
            print(f"Error in LoadImageNode: {str(e)}")
            raise


def create_graph_for_model(model_name, prompt, negative_prompt, width, height, guidance_scale, num_inference_steps,
                           batch_size, img2img_bool=False, init_image=None, controlnet_info=None):
    graph = Graph()
    model_info = get_model_info(model_name)
    print(f"model_info: {model_info}")

    if not model_info:
        raise ValueError(f"알 수 없는 모델: {model_name}")

    model_loader = ModelLoaderNode("model_loader", model_info, controlnet_info)
    text_prompt = TextPromptNode("text_prompt")

    graph.add_node(model_loader)
    graph.add_node(text_prompt)

    control_images = []
    if img2img_bool and init_image is not None and controlnet_info:
        image_loader = LoadImageNode("image_loader")
        graph.add_node(image_loader)
        image_loader.inputs["image_input"] = init_image

        for i, control in enumerate(controlnet_info):
            print(control)
            preprocessor = ControlNetPreprocessorNode(f"preprocessor_{i}", control['preprocessor'], init_image, width, height)
            graph.add_node(preprocessor)
            graph.connect(image_loader, "image", preprocessor, "image")
            control_images.append(preprocessor)

    print(control_images)
    print("-------------------------------------")

    # ImageGeneratorNode를 여기서 생성
    image_generator = ImageGeneratorNode("image_generator", model_info)
    graph.add_node(image_generator)

    graph.connect(model_loader, "pipe", image_generator, "pipe")
    graph.connect(model_loader, "strategy", image_generator, "strategy")
    graph.connect(text_prompt, "prompt", image_generator, "prompt")
    graph.connect(text_prompt, "negative_prompt", image_generator, "negative_prompt")

    if control_images:
        print(f"Connecting {len(control_images)} control image(s) to ImageGeneratorNode")
        if len(control_images) == 1:
            graph.connect(control_images[0], "processed_image", image_generator, "control_images")
        else:
            # 여러 개의 control image가 있는 경우 처리
            control_images_list = [node.outputs["processed_image"] for node in control_images]
            image_generator.inputs["control_images"] = control_images_list

    # 입력값 설정
    text_prompt.inputs["prompt"] = prompt
    text_prompt.inputs["negative_prompt"] = negative_prompt
    image_generator.inputs["width"] = width
    image_generator.inputs["height"] = height
    image_generator.inputs["guidance_scale"] = guidance_scale
    image_generator.inputs["num_inference_steps"] = num_inference_steps
    image_generator.inputs["batch_size"] = batch_size

    image_encoder = ImageEncoderNode("image_encoder")

    graph.add_node(image_encoder)
    graph.connect(image_generator, "images", image_encoder, "images")

    print("Graph creation completed. Node connections:")

    for connection in graph.connections:
        print(f"{connection[0].name}.{connection[1]} -> {connection[2].name}.{connection[3]}")

    return graph


def get_model_info(model_name):
    return AI_MODELS.get(model_name, None)

# beec (https://huggingface.co/Intel/dpt-large).
