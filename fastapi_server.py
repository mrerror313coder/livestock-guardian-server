# ============================================================
# UPDATED fastapi_server.py
# Use this exact version for Render deployment
# ============================================================

"""
Livestock Guardian — Biometric API Server
Deployed on Render.com (FREE)
"""

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

# ── APP ──
app = FastAPI(
    title="Livestock Guardian Biometric API",
    description="AI muzzle recognition for livestock identity",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API KEY SECURITY ──
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def load_valid_keys():
    """Load hashed API keys from config"""
    try:
        keys_path = os.path.join("models", "api_keys_config.json")
        with open(keys_path) as f:
            data = json.load(f)
        return {
            k["key_hash"]: k
            for k in data["keys"]
            if k["active"]
        }
    except Exception as e:
        logger.error(f"Failed to load API keys: {e}")
        return {}


def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key from request header X-API-Key"""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "API key missing",
                "fix": "Add header: X-API-Key: your_key_here"
            }
        )

    key_hash  = hashlib.sha256(api_key.encode()).hexdigest()
    valid_keys = load_valid_keys()

    if key_hash not in valid_keys:
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(
            status_code=403,
            detail={"error": "Invalid API key"}
        )

    key_info = valid_keys[key_hash]
    logger.info(f"Valid request from: {key_info['name']}")
    return key_info


# ── LOAD MODEL ON STARTUP ──
MODEL_LOADED = False
session      = None
config       = None
transform    = None
IMG_SIZE     = 128


def load_model():
    """Load ONNX model and config"""
    global MODEL_LOADED, session, config, transform, IMG_SIZE

    try:
        config_path = os.path.join("models", "model_config.json")
        model_path  = os.path.join("models", "muzzle_encoder.onnx")

        with open(config_path) as f:
            config = json.load(f)

        session  = ort.InferenceSession(model_path)
        IMG_SIZE = config["image_size"]

        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                config["normalize_mean"],
                config["normalize_std"]
            )
        ])

        MODEL_LOADED = True
        logger.info(f"Model loaded successfully!")
        logger.info(f"Image size: {IMG_SIZE} | Embedding: {config['embedding_size']}")

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        MODEL_LOADED = False


# Load when app starts
load_model()


# ── HELPER FUNCTIONS ──
def image_to_embedding(image_bytes: bytes) -> list:
    """Convert image bytes to embedding vector"""
    if not MODEL_LOADED:
        raise Exception("Model not loaded")

    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).numpy()
    emb    = session.run(
        ["embedding"],
        {"muzzle_image": tensor}
    )[0][0]

    norm = np.linalg.norm(emb)
    normalized = emb / norm if norm > 0 else emb
    return normalized.tolist()


def cosine_similarity(e1: list, e2: list) -> float:
    """Cosine similarity between two embeddings"""
    return float(np.clip(
        np.dot(np.array(e1), np.array(e2)), 0, 1
    ))


def get_prd_status(confidence: float) -> dict:
    """
    PRD-defined confidence thresholds:
    > 90%   → CONFIRMED_MATCH (Green)
    85-90%  → WARNING (Amber)
    < 85%   → NO_MATCH (Red)
    """
    if confidence > 90:
        return {
            "status": "CONFIRMED_MATCH",
            "color":  "GREEN",
            "action": "Identity confirmed"
        }
    elif confidence >= 85:
        return {
            "status": "WARNING",
            "color":  "AMBER",
            "action": "Manual verification recommended"
        }
    return {
        "status": "NO_MATCH",
        "color":  "RED",
        "action": "Re-scan required or new animal"
    }


# ════════════════════════════════════════
# PUBLIC ENDPOINTS (No API key needed)
# ════════════════════════════════════════

@app.get("/")
def root():
    """Server info"""
    return {
        "service":     "Livestock Guardian Biometric API",
        "version":     "1.0.0",
        "status":      "running",
        "model_ready": MODEL_LOADED,
        "docs":        "/docs",
        "timestamp":   datetime.now().isoformat()
    }


@app.get("/health")
def health():
    """Health check — use this to check if server is awake"""
    return {
        "status":       "healthy" if MODEL_LOADED else "model_error",
        "model_loaded": MODEL_LOADED,
        "embedding_size": config["embedding_size"] if config else 0,
        "image_size":     IMG_SIZE,
        "timestamp":      datetime.now().isoformat()
    }


# ════════════════════════════════════════
# PROTECTED ENDPOINTS (API key required)
# ════════════════════════════════════════

@app.post("/biometric/generate-embedding")
async def generate_embedding(
    file:     UploadFile = File(...),
    key_info: dict       = Depends(verify_api_key)
):
    """
    Generate 128-dim embedding for a muzzle image

    Headers required:
      X-API-Key: your_android_key_here

    Body:
      file: muzzle image (jpg/png)

    Returns:
      embedding: list of 128 floats
    """
    try:
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(400, "Empty file uploaded")

        if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(400, "File too large (max 10MB)")

        embedding = image_to_embedding(image_bytes)

        return {
            "success":        True,
            "embedding":      embedding,
            "embedding_size": len(embedding),
            "client":         key_info["name"],
            "timestamp":      datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.post("/biometric/register-muzzle")
async def register_muzzle(
    file:     UploadFile = File(...),
    key_info: dict       = Depends(verify_api_key)
):
    """
    Generate embedding for new animal registration

    After calling this:
    1. Save livestock data to Supabase 'livestock' table
    2. Save returned embedding to Supabase 'embeddings' table
    3. Link embedding to livestock via livestock_id
    """
    try:
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(400, "Empty file")

        embedding = image_to_embedding(image_bytes)

        return {
            "success":        True,
            "embedding":      embedding,
            "embedding_size": len(embedding),
            "next_steps": [
                "1. Save livestock to Supabase 'livestock' table",
                "2. Save embedding to Supabase 'embeddings' table",
                "3. Link with livestock_id"
            ],
            "timestamp": datetime.now().isoformat()
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
    """
    Match muzzle image against stored embeddings

    Body:
      file:         muzzle image
      records_json: JSON string of stored embeddings
                    [{"livestock_id": "uuid", "embedding": [...]}, ...]

    Returns:
      match result with PRD confidence status
    """
    try:
        image_bytes    = await file.read()
        query_embedding = np.array(image_to_embedding(image_bytes))
        records        = json.loads(records_json)

        if not records:
            return {
                "match_found": False,
                "message":     "Database is empty",
                "embedding":   query_embedding.tolist()
            }

        best_confidence = 0.0
        best_id         = None

        for record in records:
            stored_emb = np.array(record["embedding"])
            similarity  = cosine_similarity(
                query_embedding.tolist(),
                stored_emb.tolist()
            )
            confidence = round(similarity * 100, 2)

            if confidence > best_confidence:
                best_confidence = confidence
                best_id         = record["livestock_id"]

        status_info = get_prd_status(best_confidence)

        return {
            "match_found":  status_info["status"] != "NO_MATCH",
            "livestock_id": best_id,
            "confidence":   best_confidence,
            "status":       status_info["status"],
            "color":        status_info["color"],
            "action":       status_info["action"],
            "total_compared": len(records),
            "embedding":    query_embedding.tolist(),
            "timestamp":    datetime.now().isoformat()
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
    """
    Check if animal already registered (before new registration)
    Prevents duplicate entries — PRD requirement

    Returns:
      is_duplicate: true/false
      If duplicate: returns existing livestock_id
    """
    try:
        image_bytes     = await file.read()
        query_embedding = np.array(image_to_embedding(image_bytes))
        records         = json.loads(records_json)

        highest_confidence = 0.0

        for record in records:
            stored_emb = np.array(record["embedding"])
            similarity  = cosine_similarity(
                query_embedding.tolist(),
                stored_emb.tolist()
            )
            confidence = round(similarity * 100, 2)

            if confidence > highest_confidence:
                highest_confidence = confidence

            # PRD: >90% = confirmed duplicate
            if confidence > 90:
                return {
                    "is_duplicate":          True,
                    "duplicate_livestock_id": record["livestock_id"],
                    "confidence":            confidence,
                    "message":               "Animal already registered!",
                    "embedding":             query_embedding.tolist(),
                    "timestamp":             datetime.now().isoformat()
                }

        return {
            "is_duplicate":      False,
            "highest_confidence": highest_confidence,
            "message":           "New animal — safe to register",
            "embedding":         query_embedding.tolist(),
            "timestamp":         datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))