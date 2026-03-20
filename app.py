import os
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

import psutil
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from pydantic import BaseModel
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

LOGS_DIR = "logs"
OUTPUT_DIR = "output_videos"
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_file = os.path.join(LOGS_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("voltagepark")

USE_CUDA = torch.cuda.is_available()
device = "cuda" if USE_CUDA else "cpu"
dtype = torch.bfloat16 if USE_CUDA else torch.float32

logger.info(f"Initializing model on device={device}, dtype={dtype} ...")
pipe = CogVideoXPipeline.from_pretrained(
    "zai-org/CogVideoX-2b",
    torch_dtype=dtype,
)
pipe = pipe.to(device)
logger.info("Model loaded successfully.")

class JobStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class GenReq(BaseModel):
    prompt: str
    num_frames: int = 49
    guidance_scale: float = 4.5
    seed: Optional[int] = 42
    fps: int = 12

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus

class SystemHealth(BaseModel):
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    active_jobs: int
    completed_jobs: int

# In-memory job store (fine for an assignment; not persistent across restarts)
job_store: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Video Generation API", version="1.0.0")

def run_generation_job(job_id: str, req: GenReq) -> None:
    job_store[job_id]["status"] = JobStatus.processing
    try:
        logger.info(f"[{job_id}] Generating video for prompt: {req.prompt}")

        # Seed generator for reproducibility
        g = torch.Generator(device=device)
        if req.seed is not None:
            g.manual_seed(int(req.seed))

        out = pipe(
            prompt=req.prompt,
            num_frames=int(req.num_frames),
            guidance_scale=float(req.guidance_scale),
            generator=g,
        ).frames[0]

        video_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
        export_to_video(out, video_path, fps=int(req.fps))

        job_store[job_id]["status"] = JobStatus.completed
        job_store[job_id]["output_path"] = video_path
        logger.info(f"[{job_id}] Completed: {video_path}")

    except Exception as e:
        logger.exception(f"[{job_id}] Failed: {e}")
        job_store[job_id]["status"] = JobStatus.failed
        job_store[job_id]["error"] = str(e)

@app.post("/generate", response_model=JobResponse)
async def submit_generation(req: GenReq, background_tasks: BackgroundTasks) -> JobResponse:
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_store[job_id] = {
        "status": JobStatus.pending,
        "request": req.dict(),
        "output_path": None,
        "error": None,
    }
    background_tasks.add_task(run_generation_job, job_id, req)
    return JobResponse(job_id=job_id, status=job_store[job_id]["status"])

@app.get("/status/{job_id}", response_model=JobResponse)
async def check_status(job_id: str) -> JobResponse:
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(job_id=job_id, status=job_store[job_id]["status"])

@app.get("/video/{job_id}")
async def get_video(job_id: str) -> Response:
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]
    if job["status"] != JobStatus.completed:
        raise HTTPException(status_code=400, detail=f"Video not ready. Status: {job['status']}")

    video_path = job.get("output_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    headers = {"Content-Disposition": f'attachment; filename="{os.path.basename(video_path)}"'}
    return Response(content=video_bytes, media_type="video/mp4", headers=headers)

@app.get("/health", response_model=SystemHealth)
async def system_health() -> SystemHealth:
    cpu_percent = psutil.cpu_percent()
    mem_percent = psutil.virtual_memory().percent

    gpu_usage = 0.0
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        used = torch.cuda.memory_allocated(0)
        gpu_usage = float(used / total * 100) if total else 0.0

    active_count = sum(1 for j in job_store.values() if j["status"] == JobStatus.processing)
    completed_count = sum(1 for j in job_store.values() if j["status"] == JobStatus.completed)

    return SystemHealth(
        cpu_usage=float(cpu_percent),
        memory_usage=float(mem_percent),
        gpu_usage=float(gpu_usage),
        active_jobs=int(active_count),
        completed_jobs=int(completed_count),
    )

@app.get("/", include_in_schema=False)
async def root() -> Response:
    return Response(
        content='Visit <a href="/docs">/docs</a> for Swagger UI.',
        media_type="text/html",
    )
