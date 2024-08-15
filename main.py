from flask import Flask, request, jsonify
import logging
import os
import torch
from diffusers import StableDiffusionPipeline
import io
import base64
from PIL import Image
from dataclasses import dataclass
from typing import Dict, Any
import threading
import queue
from collections import OrderedDict

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImageGenerationConfig:
    model_path: str = "./models/checkpoints/architecturerealmix_v11.safetensors"
    default_width: int = 512
    default_height: int = 512
    min_dimension: int = 128
    max_dimension: int = 1024

class ImageGenerator:
    def __init__(self, config: ImageGenerationConfig):
        self.config = config
        self.pipe = None

    def load_model(self):
        logger.info(f"Loading custom Stable Diffusion model from {self.config.model_path}")
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found at {self.config.model_path}")

        self.pipe = StableDiffusionPipeline.from_single_file(
            self.config.model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.pipe = self.pipe.to("cuda")
        logger.info("Custom model loaded successfully.")

    def generate_image(self, prompt: str, width: int, height: int, **kwargs) -> str:
        logger.info(f"Generating image with prompt: {prompt}, width: {width}, height: {height}")
        if self.pipe is None:
            self.load_model()

        image = self.pipe(prompt, width=width, height=height, **kwargs).images[0]

        if image.width != width or image.height != height:
            image = image.resize((width, height), Image.LANCZOS)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        logger.info(f"Image generated and encoded successfully. Size: {width}x{height}")
        return img_str

class JobQueue:
    def __init__(self):
        self.queue = queue.Queue()
        self.results = OrderedDict()
        self.current_job = None
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _worker(self):
        while True:
            job_id, func, args, kwargs = self.queue.get()
            with self.lock:
                self.current_job = job_id
            try:
                result = func(*args, **kwargs)
                with self.lock:
                    self.results[job_id] = {"status": "completed", "result": result}
            except Exception as e:
                with self.lock:
                    self.results[job_id] = {"status": "failed", "error": str(e)}
            finally:
                with self.lock:
                    self.current_job = None
            self.queue.task_done()

    def enqueue(self, job_id, func, *args, **kwargs):
        with self.lock:
            self.results[job_id] = {"status": "pending"}
            queue_size = self.queue.qsize() + 1  # +1 because we're adding a new job
            logger.info(f"Job {job_id} enqueued. Current queue size: {queue_size}")
        self.queue.put((job_id, func, args, kwargs))

    def get_result(self, job_id):
        with self.lock:
            return self.results.get(job_id, {"status": "not_found"})

    def get_queue_status(self):
        with self.lock:
            total_jobs = len(self.results)
            pending_jobs = sum(1 for job in self.results.values() if job["status"] == "pending")
            current_job = self.current_job
            job_list = [{"job_id": job_id, "status": job["status"]} for job_id, job in self.results.items()]
        return {
            "total_jobs": total_jobs,
            "pending_jobs": pending_jobs,
            "current_job": current_job,
            "job_list": job_list
        }

class ServerState:
    def __init__(self, config: ImageGenerationConfig):
        self.config = config
        self.generator = ImageGenerator(config)
        self.job_queue = JobQueue()

    def validate_dimensions(self, width: int, height: int) -> bool:
        return (self.config.min_dimension <= width <= self.config.max_dimension and
                self.config.min_dimension <= height <= self.config.max_dimension)

    def enqueue_task(self, prompt: str, width: int, height: int, **kwargs) -> str:
        job_id = f"job_{len(self.job_queue.results)}"
        self.job_queue.enqueue(job_id, self.generator.generate_image, prompt, width, height, **kwargs)
        queue_status = self.job_queue.get_queue_status()
        logger.info(f"Job {job_id} enqueued. Current queue status: {queue_status['pending_jobs']} pending, {queue_status['total_jobs']} total jobs")
        return job_id

    def get_job_result(self, job_id: str) -> Dict[str, Any]:
        return self.job_queue.get_result(job_id)

    def get_queue_status(self):
        return self.job_queue.get_queue_status()


state = ServerState(ImageGenerationConfig())

@app.route('/queue_status', methods=['GET'])
def queue_status():
    status = state.get_queue_status()
    return jsonify(status), 200



@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data: Dict[str, Any] = request.json
        prompt: str = data.get('prompt')
        width: int = int(data.get('width', state.config.default_width))
        height: int = int(data.get('height', state.config.default_height))

        if not prompt:
            logger.warning("No prompt provided in the request.")
            return jsonify({"error": "No prompt provided"}), 400

        if not state.validate_dimensions(width, height):
            logger.warning(f"Invalid dimensions: {width}x{height}")
            return jsonify({"error": f"Width and height must be between {state.config.min_dimension} and {state.config.max_dimension}"}), 400

        kwargs: Dict[str, Any] = {k: v for k, v in data.items() if k not in ['prompt', 'width', 'height']}

        logger.info(f"Enqueueing image generation task with prompt: {prompt}, width: {width}, height: {height}")
        job_id: str = state.enqueue_task(prompt, width, height, **kwargs)
        queue_status = state.get_queue_status()
        return jsonify({
            "status": "enqueued",
            "job_id": job_id,
            "queue_status": {
                "pending_jobs": queue_status["pending_jobs"],
                "total_jobs": queue_status["total_jobs"]
            }
        }), 202
    except Exception as e:
        logger.error(f"Error in generate_image: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id: str):
    result = state.get_job_result(job_id)
    if result["status"] == "pending":
        return jsonify({"status": "pending"}), 202
    elif result["status"] == "failed":
        return jsonify({"error": result["error"]}), 500
    else:
        return jsonify(result), 200

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, use_reloader=False)



    # 이거복사하고 state를 따로 관리할수있는 파일만들고 예를들어 status