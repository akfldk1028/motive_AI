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
import torch
import gc
# 추가로 고려할 수 있는 개선 사항:
#
# 재시도 메커니즘: 작업 실패 시 자동으로 재시도하는 로직을 추가할 수 있습니다.
# 우선순위 큐: 중요한 작업에 높은 우선순위를 부여할 수 있습니다.
# 분산 큐: 여러 인스턴스에서 작업을 분산 처리할 수 있도록 확장할 수 있습니다.
app = Flask(__name__)
logger = get_logger()

base_url = os.environ.get('BASE_URL', "https://backend.motive.beauty/api/v1")



def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(io.BytesIO(response.content))


def process_controlnet_data(controlnet_data, model_type):
    controlnet_info = []
    for control in controlnet_data:
        control_type = control['type']
        control_info = get_controlnet_info(model_type).get(control_type)
        if control_info:
            controlnet_info.append({
                'type': control_type,
                'model_path': control_info['path'],
                'preprocessor': control_info['preprocessor']
            })
        else:
            logger.warning(f"Unknown ControlNet type: {control_type} for model type: {model_type}")
    return controlnet_info

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
            request_id, graph = self.queue.get()
            state.set_current_job(request_id)
            try:
                self.update_django_status(request_id, JobStatus.PROCESSING)
                state.set_job_status(request_id, JobStatus.PROCESSING)
                graph.process()
                result = graph.nodes[-1].outputs["encoded_images"]
                state.set_job_status(request_id, JobStatus.COMPLETED, result=result)
                self.update_django_status(request_id, JobStatus.COMPLETED)
                self.process_result(request_id, result)
            except Exception as e:
                logger.error(f"Error processing job {request_id}: {str(e)}", exc_info=True)
                state.set_job_status(request_id, JobStatus.FAILED, error=str(e))
                self.update_django_status(request_id, JobStatus.FAILED, error=str(e))
            finally:
                state.set_current_job(None)
                torch.cuda.empty_cache()
                gc.collect()  # Explicitly invoke garbage collection
            self.queue.task_done()

    def enqueue(self, graph, request_id):
        state.set_job_status(request_id, JobStatus.PENDING)
        self.update_django_status(request_id, JobStatus.PENDING)
        queue_size = self.queue.qsize() + 1
        logger.info(f"Job {request_id} enqueued. Current queue size: {queue_size}")
        self.queue.put((request_id, graph))

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

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.json
        logger.info(f"Received AIRequest data: {data}")

        prompt = data.get('prompt')
        width = int(data.get('width', 1024))
        height = int(data.get('height', 1024))
        negative_prompt = data.get('negativePrompt', '')
        cfg = float(data.get('cfg', 7.0))
        steps = int(data.get('steps', 30))
        batch_size = int(data.get('batch_size', 1))
        request_id = data.get('id')
        ai_tool_id = data.get('ai_tool')
        img2img_bool = data.get('img2img_bool', False)
        controlnet_data = data.get('controlnet', []) #controlnet': [{'id': 5, 'type': 'CANNY'}]
        file_url = data.get('file')


        # if not prompt:
        #     logger.warning("No prompt provided in the request.")
        #     return jsonify({"error": "No prompt provided"}), 400


        model_name = get_model_name_from_id(ai_tool_id)
        model_info = get_model_info(model_name)
        if not model_info:
            return jsonify({"error": f"Unknown model: {model_name}"}), 400


        model_type = 'SDXL' if model_info['type'] == 'checkpoint' and model_info.get('app') == 'SDXL' else 'SD'
        controlnet_info = process_controlnet_data(controlnet_data, model_type)

        # 이미지 로드 (img2img 또는 ControlNet용)
        init_image = None
        if img2img_bool or controlnet_info:
            if file_url:
                try:
                    init_image = load_image_from_url(file_url)
                    logger.info(f"Successfully loaded image from URL: {file_url}")
                except Exception as e:
                    logger.error(f"Failed to load image from URL: {file_url}. Error: {str(e)}")
                    return jsonify({"error": "Failed to load image from provided URL"}), 400
            else:
                logger.warning("img2img or ControlNet requested but no image URL provided.")
                return jsonify({"error": "Image URL required for img2img or ControlNet"}), 400

        try:
            graph = create_graph_for_model(
                model_name, prompt, negative_prompt, width, height, cfg, steps, batch_size,
                img2img_bool=img2img_bool, init_image=init_image, controlnet_info=controlnet_info
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        logger.info(f"Enqueueing image generation task for model: {model_name}, prompt: {prompt}, "
                    f"width: {width}, height: {height}, negative_prompt: {negative_prompt}, "
                    f"cfg: {cfg}, steps: {steps}, batch_size: {batch_size}, request_id: {request_id}")

        job_queue.enqueue(graph, request_id)

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
# ... (keep the rest of the Flask routes as they were)

def get_model_name_from_id(ai_tool_id):
    # This is a placeholder function. You need to implement the logic to map ai_tool_id to model name
    # You might want to store this mapping in a database or a configuration file
    model_mapping = {
        1: "SDXL",
        2: "Building Model",
        3: "Modern Interior",
        4: "Zach illustration",
        5: "ReRender",
        6: "Architecture RealMix",
        7: "RealisticVision",

    }
    return model_mapping.get(ai_tool_id, "SDXL")  # Default to SDXL if id not found





if __name__ == '__main__':
    logger.info(f"Starting Flask server with BASE_URL: {base_url}")
    app.run(host='0.0.0.0', port=5000)
    # app.run(host='0.0.0.0', port=5000, ssl_context=('cert.pem', 'key.pem'))


    # logger.info("Starting Flask server...")
    # app.run(debug=True, use_reloader=False)