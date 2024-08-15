import requests
import time
import base64
from PIL import Image
import io
import os
from enum import Enum

# 서버 URL
url = "http://127.0.0.1:5000"

# 이미지 저장 경로
save_directory = "./example"

class JobStatus(Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NOT_FOUND = "NOT_FOUND"

def request_image(prompt, width=512, height=512):
    try:
        response = requests.post(f"{url}/generate", json={"prompt": prompt, "width": width, "height": height},
                                 timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error requesting image: {e}")
        return None

def check_result(job_id):
    try:
        response = requests.get(f"{url}/result/{job_id}", timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error checking result: {e}")
        return None

def save_image(base64_string, filename):
    try:
        os.makedirs(save_directory, exist_ok=True)
        file_path = os.path.join(save_directory, filename)
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        image.save(file_path)
        print(f"Image saved as {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def generate_and_save_image(prompt, width=512, height=512):
    print(f"Requesting image generation with prompt: '{prompt}'")
    response = request_image(prompt, width, height)
    if response and response.get("status") == JobStatus.PENDING.name:
        job_id = response.get("job_id")
        queue_status = response.get("queue_status", {})
        print(f"Job enqueued with ID: {job_id}")
        print(f"Current queue status: {queue_status.get('pending_jobs', 'N/A')} pending, {queue_status.get('total_jobs', 'N/A')} total jobs")

        while True:
            result = check_result(job_id)
            if result is None:
                print("Failed to check job status.")
                break

            status = JobStatus(result.get("status"))
            if status == JobStatus.COMPLETED:
                image_data = result.get("result")
                if image_data:
                    file_path = save_image(image_data, f"generated_image_{int(time.time())}.png")
                    if file_path:
                        print(f"Image generation successful. File saved at: {file_path}")
                    else:
                        print("Failed to save the image. Please check the save directory.")
                else:
                    print("No image data received from the server.")
                break
            elif status == JobStatus.FAILED:
                print(f"Job failed: {result.get('error')}")
                break
            elif status == JobStatus.PENDING:
                print("Job is still processing. Waiting...")
                time.sleep(5)  # Wait for 5 seconds before checking again
            else:
                print(f"Unexpected job status: {status}")
                break
    else:
        print("Failed to enqueue image generation job. Please check the server logs for details.")

def check_queue_status():
    try:
        response = requests.get(f"{url}/queue_status", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error checking queue status: {e}")
        return None

def display_queue_status():
    status = check_queue_status()
    if status:
        print("\nCurrent Queue Status:")
        print(f"Total jobs: {status['total_jobs']}")
        print(f"Pending jobs: {status['pending_jobs']}")
        print(f"Current job: {status['current_job']}")
        print("\nJob List:")
        for job in status['job_list']:
            print(f"Job ID: {job['job_id']}, Status: {JobStatus(job['status']).name}")
    else:
        print("Failed to retrieve queue status.")

def main():
    print("Welcome to the Image Generation Service!")
    print("Enter your prompts below. Type 'quit' to exit or 'status' to check queue status.")

    while True:
        prompt = input("\nEnter a prompt for image generation (or 'status' or 'quit'): ").strip()
        if prompt.lower() == 'quit':
            break
        if prompt.lower() == 'status':
            display_queue_status()
            continue
        if not prompt:
            print("Prompt cannot be empty. Please try again.")
            continue

        width = int(input("Enter image width (default 512): ") or 512)
        height = int(input("Enter image height (default 512): ") or 512)

        generate_and_save_image(prompt, width, height)

    print("\nThank you for using the Image Generation Service. Goodbye!")

if __name__ == "__main__":
    main()