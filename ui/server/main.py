from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import subprocess
import shutil
import hashlib
import uuid
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
IMAGENET_CLASSES = []
if os.path.exists(os.path.join(BASE_DIR, "imagenet_classes.json")):
    with open(os.path.join(BASE_DIR, "imagenet_classes.json"), "r") as f:
        IMAGENET_CLASSES = json.load(f)

MODEL_CONFIGS = {
    "simple_cnn": {"path": "best_model.pth", "input_size": 32, "dataset": "cifar10"},
    "alexnet": {"path": None, "input_size": 224, "dataset": "imagenet"},
    "vgg16": {"path": None, "input_size": 224, "dataset": "imagenet"},
    "inception_v1": {"path": None, "input_size": 224, "dataset": "imagenet"}
}

tasks = {}

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

@app.get("/graph")
async def get_graph():
    metadata_path = os.path.join(RESULTS_DIR, "graph_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {"error": "Metadata not found"}

@app.get("/classes/{dataset}")
async def get_classes(dataset: str):
    if dataset == "cifar10":
        return CIFAR10_CLASSES
    elif dataset == "imagenet":
        return IMAGENET_CLASSES
    return {"error": "Unknown dataset"}

@app.get("/models")
async def get_models():
    return MODEL_CONFIGS

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    return tasks.get(task_id, {"error": "Task not found"})

async def run_generation(task_id: str, model_type: str, image_path: str, label: int, input_size: int, model_path: str, cache_path: str):
    try:
        script_path = os.path.join(BASE_DIR, "generate_all.py")
        cmd = [
            "python3", script_path, 
            "--model_type", model_type,
            "--image", image_path, 
            "--output", cache_path, 
            "--label", str(label),
            "--input_size", str(input_size)
        ]
        if model_path:
            cmd.extend(["--model_path", model_path])
            
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode().strip()
            if text.startswith("PROGRESS:"):
                try:
                    progress = int(text.split(":")[1].strip())
                    tasks[task_id]["progress"] = progress
                except:
                    pass

        await process.wait()
        
        if process.returncode != 0:
            stderr = await process.stderr.read()
            tasks[task_id]["error"] = stderr.decode()
            tasks[task_id]["status"] = "failed"
            return

        metadata_path = os.path.join(cache_path, "graph_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                tasks[task_id]["result"] = json.load(f)
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["progress"] = 100
                # Sync to results for immediate serving
                for filename in os.listdir(cache_path):
                    shutil.copy2(os.path.join(cache_path, filename), RESULTS_DIR)
        else:
            tasks[task_id]["error"] = "Generation failed to produce metadata"
            tasks[task_id]["status"] = "failed"
            
    except Exception as e:
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["status"] = "failed"

@app.post("/upload")
async def upload_and_generate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    label: int = Form(...),
    model_type: str = Form("simple_cnn")
):
    content = await file.read()
    image_hash = hashlib.md5(content).hexdigest()
    await file.seek(0)
    
    cache_key = f"{model_type}_{label}_{image_hash}"
    cache_path = os.path.join(CACHE_DIR, cache_key)
    
    if os.path.exists(os.path.join(cache_path, "graph_metadata.json")):
        for filename in os.listdir(cache_path):
            src = os.path.join(cache_path, filename)
            dst = os.path.join(RESULTS_DIR, filename)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        
        with open(os.path.join(cache_path, "graph_metadata.json"), "r") as f:
            return {"status": "completed", "result": json.load(f)}

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "progress": 0}
    
    file_ext = os.path.splitext(file.filename)[1]
    image_path = os.path.join(UPLOAD_DIR, f"{image_hash}{file_ext}")
    if not os.path.exists(image_path):
        with open(image_path, "wb") as buffer:
            buffer.write(content)
    
    config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS["simple_cnn"])
    model_path = config.get("path")
    input_size = config.get("input_size", 224)

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    background_tasks.add_task(run_generation, task_id, model_type, image_path, label, input_size, model_path, cache_path)
    
    return {"task_id": task_id, "status": "processing"}

if os.path.exists(RESULTS_DIR):
    app.mount("/images", StaticFiles(directory=RESULTS_DIR), name="images")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
