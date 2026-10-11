import os
import sys
import time
import base64
import io
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# App initialization
app = FastAPI(
    title="EcoSort Edge-AI API",
    description="Ultra-efficient waste classification with optimized MobileNet ONNX architecture",
    version="1.0.0"
)

# Enable CORS for external client applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Model Paths (checks env, root, and local model.onnx)
ONNX_MODEL_PATH = os.environ.get("MODEL_PATH", "")
if not ONNX_MODEL_PATH or not os.path.exists(ONNX_MODEL_PATH):
    for candidate in [
        os.path.join(PROJECT_ROOT, "model.onnx"),
        os.path.join(BASE_DIR, "model.onnx"),
        os.path.join(os.getcwd(), "model.onnx")
    ]:
        if os.path.exists(candidate):
            ONNX_MODEL_PATH = candidate
            break

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Normalization constants (ImageNet standards used in training)
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Comprehensive Eco Guidance Knowledge Base
ECO_GUIDANCE: Dict[str, Dict[str, Any]] = {
    "cardboard": {
        "name": "Cardboard",
        "icon": "📦",
        "theme_color": "#f59e0b",
        "bin_name": "Blue Bin (Paper & Cardboard)",
        "bin_color": "#3b82f6",
        "recyclability": "100% Recyclable",
        "badge_class": "badge-cardboard",
        "prep_steps": [
            "Flatten and break down boxes to conserve bin volume.",
            "Remove plastic packing tape, styrofoam, and bubble wrap inserts.",
            "Ensure cardboard is clean; heavily grease-stained parts belong in compost or general waste."
        ],
        "environmental_impact": "Recycling 1 ton of cardboard saves 17 mature trees, 7,000 gallons of water, and 4,000 kWh of energy.",
        "common_items": "Shipping cartons, cereal & shoe boxes, corrugated packaging, paper towel tubes."
    },
    "glass": {
        "name": "Glass Container",
        "icon": "🍾",
        "theme_color": "#14b8a6",
        "bin_name": "Green Bin (Glass Packaging)",
        "bin_color": "#10b981",
        "recyclability": "100% Infinitely Recyclable",
        "badge_class": "badge-glass",
        "prep_steps": [
            "Empty and rinse containers to remove food remnants and residues.",
            "Remove corks and metal screw-caps (metal caps can be recycled separately).",
            "Keep whole. Broken drinking glasses, window panes, and mirrors should be placed in General Trash due to different melting points."
        ],
        "environmental_impact": "Glass can be recycled endlessly with zero degradation in purity or strength. Recycled glass reduces kiln temperatures and lowers CO2 emissions by up to 30%.",
        "common_items": "Soda & beverage bottles, sauce jars, condiment bottles, cosmetic jars."
    },
    "metal": {
        "name": "Metal & Cans",
        "icon": "🥫",
        "theme_color": "#94a3b8",
        "bin_name": "Yellow / Gray Bin (Metals & Cans)",
        "bin_color": "#eab308",
        "recyclability": "100% Infinitely Recyclable",
        "badge_class": "badge-metal",
        "prep_steps": [
            "Rinse food and beverage cans clean of residue.",
            "Crush aluminum cans if convenient to optimize bin space.",
            "Paper labels do not need to be removed; they burn away cleanly in smelting furnaces."
        ],
        "environmental_impact": "Recycling a single aluminum can saves 95% of the energy needed to make a new one from raw bauxite ore—saving enough energy to power a laptop for 5 hours.",
        "common_items": "Aluminum beverage cans, tin soup/pet food cans, clean aluminum foil & trays, metal lids."
    },
    "paper": {
        "name": "Paper Waste",
        "icon": "📄",
        "theme_color": "#3b82f6",
        "bin_name": "Blue Bin (Paper Recycling)",
        "bin_color": "#2563eb",
        "recyclability": "Highly Recyclable (5-7 cycles)",
        "badge_class": "badge-paper",
        "prep_steps": [
            "Keep clean and dry; moisture and oil spoil paper fiber recycling.",
            "Remove heavy plastic sleeves, wax coatings, and metal ring binders.",
            "Staples are acceptable; industrial pulpers filter them out automatically."
        ],
        "environmental_impact": "Recycling 1 ton of paper saves 17 trees, 3 cubic yards of landfill space, and 380 gallons of oil.",
        "common_items": "Office paper, newspapers, magazines, brochures, envelopes, paper bags, notebooks."
    },
    "plastic": {
        "name": "Plastic Container",
        "icon": "🧴",
        "theme_color": "#f97316",
        "bin_name": "Yellow Bin (Plastics & Packaging)",
        "bin_color": "#f59e0b",
        "recyclability": "Widely Recyclable (PET #1, HDPE #2, PP #5)",
        "badge_class": "badge-plastic",
        "prep_steps": [
            "Empty all liquids and rinse container clean.",
            "Squash or flatten bottles to minimize bin volume.",
            "Check resin identification codes (numbers #1 and #2 have the highest recycling value)."
        ],
        "environmental_impact": "Recycling plastic conserves crude oil, reduces greenhouse emissions by 67%, and prevents harmful microplastics from polluting terrestrial and marine ecosystems.",
        "common_items": "Water bottles, detergent jugs, shampoo bottles, yogurt containers, milk jugs."
    },
    "trash": {
        "name": "Non-Recyclable Waste",
        "icon": "🗑️",
        "theme_color": "#ef4444",
        "bin_name": "Black / Dark Gray Bin (General Landfill)",
        "bin_color": "#475569",
        "recyclability": "Non-Recyclable / Residual",
        "badge_class": "badge-trash",
        "prep_steps": [
            "Bag waste securely to prevent litter and odor dispersion.",
            "Ensure no batteries, chemical solvents, or electronics are mixed in (deliver to e-waste depots).",
            "Evaluate whether single-use products can be substituted with durable, reusable alternatives in the future."
        ],
        "environmental_impact": "Residual waste sent to landfills breaks down anaerobically and produces methane gas. Diligent sorting redirects valuable raw materials away from landfills.",
        "common_items": "Soiled food wrappers, greasy chip bags, broken ceramics, dirty tissues, multi-material pouches."
    }
}

# ONNX Runtime Session
ort_session: Optional[ort.InferenceSession] = None
model_meta: Dict[str, Any] = {}

def get_session() -> ort.InferenceSession:
    global ort_session, model_meta
    if ort_session is None:
        if not os.path.exists(ONNX_MODEL_PATH):
            raise RuntimeError(f"ONNX model file not found at {ONNX_MODEL_PATH}")
        
        # Load ONNX Session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH, opts, providers=["CPUExecutionProvider"])
        
        file_size_mb = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
        input_info = ort_session.get_inputs()[0]
        output_info = ort_session.get_outputs()[0]
        
        # Parameter counting
        try:
            import onnx
            model_def = onnx.load(ONNX_MODEL_PATH)
            param_count = sum(np.prod(init.dims) for init in model_def.graph.initializer)
        except Exception:
            param_count = 324582

        model_meta = {
            "model_path": ONNX_MODEL_PATH,
            "file_size_mb": round(file_size_mb, 2),
            "total_parameters": int(param_count),
            "input_name": input_info.name,
            "input_shape": input_info.shape,
            "output_name": output_info.name,
            "output_shape": output_info.shape,
            "classes": CLASSES
        }
    return ort_session

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess PIL image strictly following EcoSort specifications (96x96 RGB, ImageNet normalized)."""
    image = image.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    # Shape: (3, 96, 96)
    arr = np.transpose(arr, (2, 0, 1))
    # Normalize
    arr = (arr - NORM_MEAN) / NORM_STD
    # Batch dimension: (1, 3, 96, 96)
    return np.expand_dims(arr, axis=0)

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

class Base64Payload(BaseModel):
    image_base64: str

@app.on_event("startup")
def startup_event():
    try:
        get_session()
        print(f"[EcoSort] Model loaded successfully: {model_meta['file_size_mb']} MB, {model_meta['total_parameters']:,} parameters")
    except Exception as e:
        print(f"[EcoSort] Error loading ONNX model on startup: {e}")

@app.get("/api/health")
def health_check():
    session = get_session()
    return {
        "status": "healthy",
        "engine": "ONNX Runtime (CPUExecutionProvider)",
        "metadata": model_meta
    }

@app.get("/api/classes")
def get_classes():
    return {
        "classes": CLASSES,
        "guidance": ECO_GUIDANCE
    }

@app.get("/api/samples")
def get_samples():
    """Returns accessible sample images for instant 1-click test drive."""
    samples = []
    for cls in CLASSES:
        samples.append({
            "class": cls,
            "name": ECO_GUIDANCE[cls]["name"],
            "icon": ECO_GUIDANCE[cls]["icon"],
            "url": f"/static/samples/{cls}.png"
        })
    return {"samples": samples}

def run_prediction(image: Image.Image) -> Dict[str, Any]:
    session = get_session()
    
    t0 = time.perf_counter()
    tensor = preprocess_image(image)
    
    t_inf_start = time.perf_counter()
    outputs = session.run([model_meta["output_name"]], {model_meta["input_name"]: tensor})[0][0]
    latency_ms = (time.perf_counter() - t_inf_start) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    
    probabilities = softmax(outputs)
    pred_idx = int(np.argmax(probabilities))
    pred_class = CLASSES[pred_idx]
    confidence = float(probabilities[pred_idx] * 100.0)
    
    all_scores = [
        {
            "class": c,
            "name": ECO_GUIDANCE[c]["name"],
            "icon": ECO_GUIDANCE[c]["icon"],
            "theme_color": ECO_GUIDANCE[c]["theme_color"],
            "percentage": round(float(probabilities[i] * 100.0), 2)
        }
        for i, c in enumerate(CLASSES)
    ]
    # Sort descending
    all_scores.sort(key=lambda x: x["percentage"], reverse=True)
    
    return {
        "status": "success",
        "predicted_class": pred_class,
        "confidence": round(confidence, 2),
        "latency_ms": round(latency_ms, 2),
        "total_time_ms": round(total_ms, 2),
        "probabilities": all_scores,
        "guidance": ECO_GUIDANCE[pred_class]
    }

@app.post("/api/predict")
async def predict_file(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None)
):
    """
    Accepts either multipart file upload or base64 data string (from webcam or client).
    """
    image_data: Optional[bytes] = None
    
    if file is not None:
        image_data = await file.read()
    elif image_base64 is not None and image_base64.strip():
        b64_str = image_base64.strip()
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        try:
            image_data = base64.b64decode(b64_str)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
            
    if not image_data:
        raise HTTPException(status_code=400, detail="No image provided. Please upload a file or base64 image.")

    try:
        pil_image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image file: {str(e)}")

    result = run_prediction(pil_image)
    return JSONResponse(content=result)

@app.post("/api/predict-json")
async def predict_json(payload: Base64Payload):
    """JSON-based predict endpoint for client applications and IoT devices."""
    b64_str = payload.image_base64.strip()
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    try:
        image_data = base64.b64decode(b64_str)
        pil_image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")
        
    result = run_prediction(pil_image)
    return JSONResponse(content=result)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "EcoSort Edge-AI API is running. Index file not found."}
