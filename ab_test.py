from flask import Flask, request, jsonify
import threading
import queue
from collections import OrderedDict
import os
import requests
from state import state, JobStatus, get_logger
from node_system import create_graph_for_model
import base64
from config import get_model_info, get_controlnet_info
from PIL import Image
import io
import itertools

app = Flask(__name__)
logger = get_logger()

base_url = "http://127.0.0.1:8000/api/v1"


def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(io.BytesIO(response.content))


def process_controlnet_data(controlnet_data, model_type):


# 기존 코드와 동일

def get_upload_url():


# 기존 코드와 동일

def upload_image_to_cloudflare(filename, upload_url, img_bytes):


# 기존 코드와 동일

class ABTestJobQueue(JobQueue):
    def _worker(self):
        while True:
            job = self.queue.get()
            request_id, test_type, graph = job
            state.set_current_job(request_id)
            try:
                self.update_django_status(request_id, JobStatus.PROCESSING)
                state.set_job_status(request_id, JobStatus.PROCESSING)

                graph.process()
                result = graph.nodes[-1].outputs["encoded_images"]

                state.set_job_status(request_id, JobStatus.COMPLETED, result=result)
                self.update_django_status(request_id, JobStatus.COMPLETED)
                self.process_result(request_id, result, test_type)
            except Exception as e:
                logger.error(f"Error processing job {request_id}: {str(e)}", exc_info=True)
                state.set_job_status(request_id, JobStatus.FAILED, error=str(e))
                self.update_django_status(request_id, JobStatus.FAILED, error=str(e))
            finally:
                state.set_current_job(None)
            self.queue.task_done()


    def process_result(self, request_id, results, test_type):
        try:
            output_image_urls = []
            for i, result_set in enumerate(results):
                for j, image_base64 in enumerate(result_set):
                    upload_url = get_upload_url()
                    cloudflare_url = upload_url["uploadURL"]
                    img_data = base64.b64decode(image_base64)
                    filename = f"ab_test_{test_type}_{request_id}_{i}_{j}.png"
                    cloudflare_result = upload_image_to_cloudflare(filename, cloudflare_url, img_data)
                    output_image_urls.append(cloudflare_result['result']['variants'][0])

            data = {
                "request_id": request_id,
                "output_image_urls": output_image_urls,
                "status": JobStatus.COMPLETED.name,
                "test_type": test_type
            }
            self.send_result_to_django(data)
            self.update_django_status(request_id, JobStatus.COMPLETED)

        except Exception as e:
            logger.error(f"Error processing AB test result for job {request_id}: {str(e)}", exc_info=True)
            state.set_job_status(request_id, JobStatus.FAILED, error=str(e))
            self.update_django_status(request_id, JobStatus.FAILED, error=str(e))


ab_test_job_queue = ABTestJobQueue()


@app.route('/grid_test', methods=['POST'])
def grid_test():
    try:
        data = request.json
        logger.info(f"Received grid test request data: {data}")

        request_id = data.get('id')
        models = [get_model_name_from_id(model_id) for model_id in data.get('model_ids', [])]
        prompts = data.get('prompts', [])

        config = {
            'models': models,
            'prompts': prompts,
            'negative_prompt': data.get('negative_prompt', ''),
            'width': int(data.get('width', 512)),
            'height': int(data.get('height', 512)),
            'guidance_scale': float(data.get('cfg', 7.0)),
            'num_inference_steps': int(data.get('steps', 30)),
            'controlnet_info': process_controlnet_data(data.get('controlnet', []), 'SD'),
            'init_image': load_image_from_url(data['file']) if data.get('file') else None
        }

        graph = create_grid_graph(**config)

        ab_test_job_queue.enqueue((request_id, 'grid', graph))

        queue_status = state.get_queue_status()
        return jsonify({
            "status": JobStatus.PENDING.name,
            "request_id": request_id,
            "queue_status": {
                "pending_jobs": queue_status["pending_jobs"],
                "total_jobs": queue_status["total_jobs"]
            }
        }), 202

    except Exception as e:
        logger.error(f"Error in grid_test: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == '__main__':
    logger.info("Starting AB Test Flask server...")
    app.run(debug=True, use_reloader=False, port=5001)


class GridImageNode(Node):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "images": [],
            "models": [],
            "prompts": [],
            "cell_width": 512,
            "cell_height": 512,
        }
        self.outputs = {"grid_image": None}

    def process(self):
        images = self.inputs["images"]
        models = self.inputs["models"]
        prompts = self.inputs["prompts"]
        cell_width = self.inputs["cell_width"]
        cell_height = self.inputs["cell_height"]

        num_models = len(models)
        num_prompts = len(prompts)

        # 그리드 이미지 크기 계산 (레이블 공간 포함)
        grid_width = (num_models + 1) * cell_width
        grid_height = (num_prompts + 1) * cell_height
        from PIL import Image, ImageDraw, ImageFont

        # 새 이미지 생성
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

        draw = ImageDraw.Draw(grid_image)
        font = ImageFont.load_default()

        # 모델 이름 쓰기
        for i, model in enumerate(models):
            draw.text((cell_width * (i + 1), 10), model, font=font, fill='black')

        # 프롬프트 쓰기
        for j, prompt in enumerate(prompts):
            draw.text((10, cell_height * (j + 1)), prompt[:20], font=font, fill='black')

        # 이미지 붙이기
        for i, model in enumerate(models):
            for j, prompt in enumerate(prompts):
                img_index = i * num_prompts + j
                if img_index < len(images):
                    img = images[img_index].resize((cell_width, cell_height))
                    grid_image.paste(img, ((i + 1) * cell_width, (j + 1) * cell_height))

        self.outputs["grid_image"] = grid_image


def create_grid_graph(models, prompts, negative_prompt, width, height, guidance_scale, num_inference_steps,
                      controlnet_info=None, init_image=None):
    graph = Graph()

    grid_generator = GridGeneratorNode("grid_generator")
    grid_generator.inputs["models"] = models
    grid_generator.inputs["prompts"] = prompts
    grid_generator.inputs["negative_prompt"] = negative_prompt
    grid_generator.inputs["width"] = width
    grid_generator.inputs["height"] = height
    grid_generator.inputs["guidance_scale"] = guidance_scale
    grid_generator.inputs["num_inference_steps"] = num_inference_steps
    grid_generator.inputs["controlnet_info"] = controlnet_info
    grid_generator.inputs["init_image"] = init_image

    graph.add_node(grid_generator)

    grid_image = GridImageNode("grid_image")
    grid_image.inputs["cell_width"] = width
    grid_image.inputs["cell_height"] = height
    grid_image.inputs["models"] = models
    grid_image.inputs["prompts"] = prompts

    graph.add_node(grid_image)
    graph.connect(grid_generator, "images", grid_image, "images")

    image_encoder = ImageEncoderNode("image_encoder")
    graph.add_node(image_encoder)
    graph.connect(grid_image, "grid_image", image_encoder, "images")

    return graph