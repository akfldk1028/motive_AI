from flask import Flask, request, jsonify
import torch
from diffusers import StableDiffusionPipeline
import io
import base64
from PIL import Image
import threading
import queue
from collections import OrderedDict
import os
import requests
from state import state, JobStatus, get_logger

app = Flask(__name__)
logger = get_logger()

base_url = "http://127.0.0.1:8000/api/v1"

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

class ImageGenerator:
    def __init__(self):
        self.pipe = None

    def load_model(self):
        logger.info(f"Loading custom Stable Diffusion model from {state.config.model_path}")
        if not os.path.exists(state.config.model_path):
            raise FileNotFoundError(f"Model file not found at {state.config.model_path}")

        self.pipe = StableDiffusionPipeline.from_single_file(
            state.config.model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.pipe = self.pipe.to("cuda")
        logger.info("Custom model loaded successfully.")

    def generate_image(self, prompt: str, width: int, height: int, negative_prompt: str = None,
                       guidance_scale: float = 7.0, num_inference_steps: int = 30,
                       batch_size: int = 1, **kwargs) -> list:
        logger.info(f"Generating {batch_size} image(s) with prompt: {prompt}, width: {width}, height: {height}, "
                    f"negative_prompt: {negative_prompt}, guidance_scale: {guidance_scale}, "
                    f"num_inference_steps: {num_inference_steps}")

        if self.pipe is None:
            self.load_model()

        images = self.pipe(
            [prompt] * batch_size,
            width=width,
            height=height,
            negative_prompt=[negative_prompt] * batch_size if negative_prompt else None,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            **kwargs
        ).images

        return [image_to_base64(img) for img in images]  # Convert images to base64

def get_upload_url():
    url = f"{base_url}/medias/photos/get-url"
    response = requests.post(url)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

def upload_image_to_cloudflare(filename, upload_url, img_bytes):
    files = {'file': (filename, img_bytes, 'image/png')}
    response = requests.post(upload_url, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

class JobQueue:
    def __init__(self):
        self.queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _worker(self):
        while True:
            request_id, func, args, kwargs = self.queue.get()
            state.set_current_job(request_id)
            try:
                self.update_django_status(request_id, JobStatus.PROCESSING)
                state.set_job_status(request_id, JobStatus.PROCESSING)
                result = func(*args, **kwargs)
                state.set_job_status(request_id, JobStatus.COMPLETED, result=result)
                self.update_django_status(request_id, JobStatus.COMPLETED)
                self.process_result(request_id, result)
            except Exception as e:
                logger.error(f"Error processing job {request_id}: {str(e)}", exc_info=True)
                state.set_job_status(request_id, JobStatus.FAILED, error=str(e))
                self.update_django_status(request_id, JobStatus.FAILED, error=str(e))
            finally:
                state.set_current_job(None)
            self.queue.task_done()

    def enqueue(self, func, *args, **kwargs):
        request_id = kwargs.get('request_id')
        state.set_job_status(request_id, JobStatus.PENDING)
        self.update_django_status(request_id, JobStatus.PENDING)
        queue_size = self.queue.qsize() + 1
        logger.info(f"Job {request_id} enqueued. Current queue size: {queue_size}")
        self.queue.put((request_id, func, args, kwargs))

    def process_result(self, request_id, images):
        try:
            output_image_urls = []

            for i, image_base64 in enumerate(images):
                # Get a new upload URL for each image
                upload_url = get_upload_url()
                cloudflare_url = upload_url["uploadURL"]

                img_data = base64.b64decode(image_base64)
                filename = f"generated_image_{request_id}_{i}.png"
                cloudflare_result = upload_image_to_cloudflare(filename, cloudflare_url, img_data)
                output_image_urls.append(cloudflare_result['result']['variants'][0])

            data = {
                "request_id": request_id,
                "output_image_urls": output_image_urls,
                "status": JobStatus.COMPLETED.name
            }
            self.send_result_to_django(data)
            self.update_django_status(request_id, JobStatus.COMPLETED)

        except Exception as e:
            logger.error(f"Error processing result for job {request_id}: {str(e)}", exc_info=True)
            state.set_job_status(request_id, JobStatus.FAILED, error=str(e))
            self.update_django_status(request_id, JobStatus.FAILED, error=str(e))


    def send_result_to_django(self, data):
        try:
            csrf_response = requests.get(f"{base_url}/airequests/get-csrf-token")
            csrf_token = csrf_response.json().get('csrfToken')

            headers = {
                'X-CSRFToken': csrf_token,
                'Content-Type': 'application/json'
            }
            adjusted_data = {
                "request_id": data["request_id"],
                "output_images": data["output_image_urls"]
            }
            url = f"{base_url}/airequests/response"
            response = requests.post(url, headers=headers, json=adjusted_data)
            response.raise_for_status()
            logger.info(f"Result sent to Django server successfully for job {data['request_id']}")
        except requests.RequestException as e:
            logger.error(f"Failed to send data to Django server: {str(e)}", exc_info=True)

    def update_django_status(self, request_id, status, error=None):
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            data = {
                "request_id": request_id,
                "status": status.name,
                "error": error
            }
            url = f"{base_url}/airequests/update-status"
            logger.info(f"Sending status update to Django: {data}")
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            logger.info(f"Status updated in Django server for request {request_id}: {status.name}")
        except requests.RequestException as e:
            logger.error(f"Failed to update status in Django server: {str(e)}")
            logger.error(f"Response content: {e.response.content if e.response else 'No response content'}")

job_queue = JobQueue()
generator = ImageGenerator()

@app.route('/queue_status', methods=['GET'])
def queue_status():
    status = state.get_queue_status()
    return jsonify(status), 200

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.json
        logger.info(f"Received AIRequest data: {data}")

        prompt = data.get('prompt')
        width = int(data.get('width', state.config.default_width))
        height = int(data.get('height', state.config.default_height))
        negative_prompt = data.get('negativePrompt')
        cfg = float(data.get('cfg', 7.0))
        steps = int(data.get('steps', 30))
        batch_size = int(data.get('batch_size', 1))
        request_id = data.get('id')

        if not prompt:
            logger.warning("No prompt provided in the request.")
            return jsonify({"error": "No prompt provided"}), 400

        if not state.validate_dimensions(width, height):
            logger.warning(f"Invalid dimensions: {width}x{height}")
            return jsonify({
                "error": f"Width and height must be between {state.config.min_dimension} and {state.config.max_dimension}"
            }), 400

        kwargs = {
            "negative_prompt": negative_prompt,
            "guidance_scale": cfg,
            "num_inference_steps": steps,
            "batch_size": batch_size,
            "request_id": request_id,
        }

        logger.info(f"Enqueueing image generation task with prompt: {prompt}, width: {width}, height: {height}, "
                    f"negative_prompt: {negative_prompt}, cfg: {cfg}, steps: {steps}, batch_size: {batch_size}, "
                    f"request_id: {request_id}")

        job_queue.enqueue(generator.generate_image, prompt, width, height, **kwargs)

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
        logger.error(f"Error in generate_image: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/result/<request_id>', methods=['GET'])
def get_result(request_id: str):
    result = state.get_job_result(request_id)
    status_code = 200
    if result["status"] == JobStatus.PENDING:
        status_code = 202
    elif result["status"] == JobStatus.PROCESSING:
        status_code = 202
    elif result["status"] == JobStatus.FAILED:
        status_code = 500

    return jsonify({
        "status": result["status"].name,
        "message": result.get("message", ""),
        "error": result.get("error", ""),
        "result": result.get("result", "")
    }), status_code

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, use_reloader=False)
    #TODO 로그 찍히게 몇번째 QUEUE 인지
