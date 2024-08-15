from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging
from collections import OrderedDict

class JobStatus(Enum):
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    NOT_FOUND = auto()

@dataclass
class ImageGenerationConfig:
    model_path: str = "./models/checkpoints/architecturerealmix_v11.safetensors"
    default_width: int = 512
    default_height: int = 512
    min_dimension: int = 128
    max_dimension: int = 1024

class ServerState:
    def __init__(self, config: ImageGenerationConfig):
        self.config = config
        self.job_results: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.current_job: Optional[str] = None
        self.completed_jobs: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_completed_jobs = 100  # 완료된 작업의 최대 저장 개수

    def set_job_status(self, request_id: str, status: JobStatus, result: Any = None, error: str = None):
        if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            if request_id in self.job_results:
                job = self.job_results.pop(request_id)
                job["status"] = status
                job["result"] = result
                job["error"] = error
                self.completed_jobs[request_id] = job
                if len(self.completed_jobs) > self.max_completed_jobs:
                    self.completed_jobs.popitem(last=False)
        else:
            self.job_results[request_id] = {
                "status": status,
                "result": result,
                "error": error
            }

    def get_job_result(self, request_id: str) -> Dict[str, Any]:
        if request_id in self.job_results:
            return self.job_results[request_id]
        elif request_id in self.completed_jobs:
            return self.completed_jobs[request_id]
        else:
            return {"status": JobStatus.NOT_FOUND}

    def set_current_job(self, request_id: Optional[str]):
        self.current_job = request_id

    def get_queue_status(self):
        total_jobs = len(self.job_results)
        pending_jobs = sum(1 for job in self.job_results.values() if job["status"] == JobStatus.PENDING)
        job_list = [{"request_id": request_id, "status": job["status"]} for request_id, job in self.job_results.items()]
        return {
            "total_jobs": total_jobs,
            "pending_jobs": pending_jobs,
            "current_job": self.current_job,
            "job_list": job_list
        }

    def validate_dimensions(self, width: int, height: int) -> bool:
        return (self.config.min_dimension <= width <= self.config.max_dimension and
                self.config.min_dimension <= height <= self.config.max_dimension)

state = ServerState(ImageGenerationConfig())

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger