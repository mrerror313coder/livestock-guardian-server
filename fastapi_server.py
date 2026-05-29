# ============================================================
# COMPLETE fastapi_server.py — FINAL VERSION
# With auto model download from Hugging Face
# Ready for Render.com deployment
# ============================================================

from fastapi import (FastAPI, UploadFile, File,
                     HTTPException, Form, Depends, Security)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import numpy as np
import json
import io
import os
import hashlib
import logging
import urllib.request
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as transforms
from datetime import datetime

# ── LOGGING ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("livestock_guardian")

# ── HUGGING FACE CONFIG ──
# Set these as Environment Variables in Render dashboard
HF_USERNAME = os.environ.get("HF_USERNAME", "YOUR_USERNAME")
HF_REPO     = os.environ.get("HF_REPO", "livestock-guardian-muzzle-model")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")  # For private repos

MODELS_DIR  = "models"

MODEL_FILES = {
    "muzzle_encoder.onnx": (
        f"https://huggingface.co/{HF_USERNAME}/{HF_REPO}"
        f"/resolve/main/muzzle_encoder.onnx"
    ),
    "model_config.json": (
        f"https://huggingface.co/{HF_USERNAME}/{HF_REPO}"
        f"/resolve/main/model_config.json"
    ),
    "api_keys_config.json": (
        f"https://huggingface.co/{HF_USERNAME}/{HF_REPO}"
        f"/resolve/main/api_keys_config.json"
    ),
}


def download_all_models():
    """Download model files from Hugging Face"""
    os.makedirs(MODELS_DIR, exist_ok=True)

    for filename, url in MODEL_FILES.items():
        filepath = os.path.join(MODELS_DIR, filename)

        if os.path.exists(filepath):
            logger.info(f"Exists: {filename}")
            continue

        logger.info(f"Downloading: {filename}...")

        try:
            headers = {}
            if HF_TOKEN:
                headers["Authorization"] = f"Bearer {HF_TOKEN}"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp:
                with open(filepath, 'wb') as f:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)

            size = os.path.getsize(filepath) / 1e6
            logger.info(f"Downloaded: {filename} ({size:.1f} MB)")

        except Exception as e:
            logger.error(f"Download failed for {filename}: {e}")
            raise RuntimeError(f"Model download failed: {e}")


# Download on startup
download_all_models()

# ── APP ──
app = FastAPI(
    title="Livestock Guardian Biometric API",
    description="AI muzzle recognition for livestock identity",
    version="1.0.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API KEY ──
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def load_valid_keys():
    try:
        with open(os.path.join(MODELS_DIR, "api_keys_config.json")) as f:
            data = json.load(f)
        return {k["key_hash"]: k for k in data["keys"] if k["active"]}
    except Exception as e:
        logger.error(f"Keys load error: {e}")
        return {}


def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(401, "API key missing. Add X-API-Key header.")
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    valid    = load_valid_keys()
    if key_hash not in valid:
        raise HTTPException(403, "Invalid API key.")
    logger.info(f"Request from: {valid[key_hash]['name']}")
    return valid[key_hash]


# ── LOAD MODEL ──
with open(os.path.join(MODELS_DIR, "model_config.json")) as f:
    config = json.load(f)

session  = ort.InferenceSession(
    os.path.join(MODELS_DIR, "muzzle_encoder.onnx")
)
IMG_SIZE = config["image_size"]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        config["normalize_mean"],
        config["normalize_std"]
    )
])

logger.info(f"Model ready! Size:{IMG_SIZE} Emb:{config['embedding_size']}")


# ── HELPERS ──
def image_to_embedding(image_bytes: bytes) -> list:
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).numpy()
    emb    = session.run(["embedding"], {"muzzle_image": tensor})[0][0]
    norm   = np.linalg.norm(emb)
    return (emb / norm if norm > 0 else emb).tolist()


def cos_sim(e1, e2) -> float:
    return float(np.clip(np.dot(np.array(e1), np.array(e2)), 0, 1))


def prd_status(conf: float) -> dict:
    if conf > 90:
        return {"status": "CONFIRMED_MATCH", "color": "GREEN",
                "action": "Identity confirmed"}
    elif conf >= 85:
        return {"status": "WARNING", "color": "AMBER",
                "action": "Manual verification recommended"}
    return {"status": "NO_MATCH", "color": "RED",
            "action": "Re-scan required"}


# ── PUBLIC ENDPOINTS ──
@app.get("/")
def root():
    return {
        "service":   "Livestock Guardian Biometric API",
        "version":   "1.0.0",
        "status":    "running",
        "docs":      "/docs",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
def health():
    return {
        "status":         "healthy",
        "model_loaded":   True,
        "embedding_size": config["embedding_size"],
        "image_size":     IMG_SIZE,
        "timestamp":      datetime.now().isoformat()
    }


# ── PROTECTED ENDPOINTS ──
@app.post("/biometric/generate-embedding")
async def generate_embedding(
    file:     UploadFile = File(...),
    key_info: dict       = Depends(verify_api_key)
):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(400, "Empty file")
        emb = image_to_embedding(data)
        return {
            "success":        True,
            "embedding":      emb,
            "embedding_size": len(emb),
            "timestamp":      datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/biometric/register-muzzle")
async def register_muzzle(
    file:     UploadFile = File(...),
    key_info: dict       = Depends(verify_api_key)
):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(400, "Empty file")
        emb = image_to_embedding(data)
        return {
            "success":        True,
            "embedding":      emb,
            "embedding_size": len(emb),
            "timestamp":      datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/biometric/match-muzzle")
async def match_muzzle(
    file:         UploadFile = File(...),
    records_json: str        = Form(...),
    key_info:     dict       = Depends(verify_api_key)
):
    try:
        query   = np.array(image_to_embedding(await file.read()))
        records = json.loads(records_json)

        if not records:
            return {"match_found": False, "message": "Empty database",
                    "embedding": query.tolist()}

        best_conf, best_id = 0.0, None
        for r in records:
            conf = round(cos_sim(query.tolist(),
                                 np.array(r["embedding"]).tolist()) * 100, 2)
            if conf > best_conf:
                best_conf, best_id = conf, r["livestock_id"]

        status = prd_status(best_conf)
        return {
            "match_found":    status["status"] != "NO_MATCH",
            "livestock_id":   best_id,
            "confidence":     best_conf,
            "status":         status["status"],
            "color":          status["color"],
            "action":         status["action"],
            "total_compared": len(records),
            "embedding":      query.tolist(),
            "timestamp":      datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/biometric/check-duplicate")
async def check_duplicate(
    file:         UploadFile = File(...),
    records_json: str        = Form(...),
    key_info:     dict       = Depends(verify_api_key)
):
    try:
        query   = np.array(image_to_embedding(await file.read()))
        records = json.loads(records_json)
        highest = 0.0

        for r in records:
            conf = round(cos_sim(query.tolist(),
                                 np.array(r["embedding"]).tolist()) * 100, 2)
            if conf > highest:
                highest = conf
            if conf > 90:
                return {
                    "is_duplicate":          True,
                    "duplicate_livestock_id": r["livestock_id"],
                    "confidence":            conf,
                    "embedding":             query.tolist(),
                    "timestamp":             datetime.now().isoformat()
                }

        return {
            "is_duplicate":       False,
            "highest_confidence": highest,
            "embedding":          query.tolist(),
            "timestamp":          datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
