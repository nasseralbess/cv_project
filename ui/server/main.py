from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import subprocess
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.get("/graph")
async def get_graph():
    metadata_path = os.path.join(RESULTS_DIR, "graph_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {"error": "Metadata not found"}

@app.post("/upload")
async def upload_and_generate(file: UploadFile = File(...), label: int = Form(...)):
    image_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        script_path = os.path.join(BASE_DIR, "generate_all.py")
        cmd = ["python3", script_path, "--image", image_path, "--output", RESULTS_DIR, "--label", str(label)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if "VALIDATION_FAILED" in result.stdout:
            # Extract prediction for better error message
            # VALIDATION_FAILED: Model predicted 3, but ground truth is 6
            prediction_idx = result.stdout.split("Model predicted ")[1].split(",")[0]
            prediction_name = CIFAR10_CLASSES[int(prediction_idx)]
            return {"error": f"Validation failed! Model predicted '{prediction_name}', but you said it's a '{CIFAR10_CLASSES[label]}'. Please upload a clearer image or check your label."}

        metadata_path = os.path.join(RESULTS_DIR, "graph_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {"error": "Generation failed to produce metadata"}
    except subprocess.CalledProcessError as e:
        return {"error": f"Script failed: {e.stderr}"}
    except Exception as e:
        return {"error": str(e)}

if os.path.exists(RESULTS_DIR):
    app.mount("/images", StaticFiles(directory=RESULTS_DIR), name="images")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
