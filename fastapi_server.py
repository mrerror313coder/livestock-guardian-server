from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException
from fastapi import Form
from fastapi import Depends
from fastapi import Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import numpy as np
import json
import io
import os
import hashlib
import logging
import urllib.request
import urllib.error
from PIL import Image
import onnxruntime as ort
from datetime import datetime

# ── LOGGING SETUP ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("lg")

# ── READ ENVIRONMENT VARIABLES ──
HF_USERNAME = os.environ.get("HF_USERNAME", "")
HF_REPO     = os.environ.get("HF_REPO", "livestock-guardian-model")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
PORT        = int(os.environ.get("PORT", 8000))
MODELS_DIR  = "models"

logger.info("=== LIVESTOCK GUARDIAN SERVER STARTING ===")
logger.info(f"HF_USERNAME = {repr(HF_USERNAME)}")
logger.info(f"HF_REPO     = {repr(HF_REPO)}")
logger.info(f"HF_TOKEN    = {'SET' if HF_TOKEN else 'NOT SET'}")


# ── STEP 1: CREATE MODELS FOLDER ──
os.makedirs(MODELS_DIR, exist_ok=True)


# ── STEP 2: DOWNLOAD FILES FROM HUGGING FACE ──
def make_url(filename):
    url = f"https://huggingface.co/{HF_USERNAME}/{HF_REPO}/resolve/main/{filename}"
    logger.info(f"URL for {filename}: {url}")
    return url


def download_single_file(filename):
    save_path = os.path.join(MODELS_DIR, filename)

    # Skip if already downloaded
    if os.path.exists(save_path):
        size = os.path.getsize(save_path)
        logger.info(f"ALREADY EXISTS: {filename} ({size} bytes)")
        return True

    if not HF_USERNAME:
        logger.error("CANNOT DOWNLOAD: HF_USERNAME is empty")
        logger.error("Go to Render → Environment → Add HF_USERNAME")
        return False

    url = make_url(filename)
    logger.info(f"DOWNLOADING: {filename}")

    request_headers = {}
    if HF_TOKEN:
        request_headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        req      = urllib.request.Request(url, headers=request_headers)
        response = urllib.request.urlopen(req, timeout=300)

        total_bytes = 0
        with open(save_path, "wb") as output_file:
            while True:
                piece = response.read(65536)
                if not piece:
                    break
                output_file.write(piece)
                total_bytes += len(piece)
                if total_bytes % (5 * 1024 * 1024) == 0:
                    logger.info(f"  {filename}: {total_bytes // (1024*1024)} MB so far")

        final_size = os.path.getsize(save_path)
        logger.info(f"DOWNLOADED OK: {filename} ({final_size} bytes)")
        return True

    except urllib.error.HTTPError as http_err:
        logger.error(f"HTTP ERROR {http_err.code} for {filename}")
        logger.error(f"URL was: {url}")
        logger.error(f"Reason: {http_err.reason}")
        logger.error("POSSIBLE CAUSES:")
        logger.error("  1. Wrong HF_USERNAME in Render environment")
        logger.error("  2. Wrong HF_REPO in Render environment")
        logger.error(f"  3. File '{filename}' not uploaded to Hugging Face")
        logger.error("  4. Repo is PRIVATE but no HF_TOKEN set")
        logger.error(f"TEST THIS URL IN BROWSER: {url}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

    except Exception as other_err:
        logger.error(f"OTHER ERROR for {filename}: {other_err}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


# Download all 3 required files
FILES_NEEDED = [
    "model_config.json",
    "api_keys_config.json",
    "muzzle_encoder.onnx",
]

logger.info("--- STARTING FILE DOWNLOADS ---")
download_results = {}
for each_file in FILES_NEEDED:
    result = download_single_file(each_file)
    download_results[each_file] = result
    logger.info(f"  {each_file}: {'OK' if result else 'FAILED'}")

all_files_ok = all(download_results.values())
logger.info(f"All files ready: {all_files_ok}")
logger.info("--- FILE DOWNLOADS DONE ---")


# ── STEP 3: CREATE FASTAPI APP ──
app = FastAPI(
    title="Livestock Guardian Biometric API",
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


# ── STEP 4: API KEY SYSTEM ──
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def read_valid_keys():
    keys_file = os.path.join(MODELS_DIR, "api_keys_config.json")
    if not os.path.exists(keys_file):
        logger.warning("api_keys_config.json not found")
        return {}
    try:
        with open(keys_file) as f:
            data = json.load(f)
        result = {}
        for k in data.get("keys", []):
            if k.get("active", False):
                result[k["key_hash"]] = k
        return result
    except Exception as err:
        logger.error(f"Cannot read keys file: {err}")
        return {}


def check_api_key(incoming_key: str = Security(api_key_header_scheme)):
    if not incoming_key:
        raise HTTPException(
            status_code=401,
            detail="No API key. Add header: X-API-Key: your_key"
        )
    hashed       = hashlib.sha256(incoming_key.encode()).hexdigest()
    valid_keys   = read_valid_keys()
    if hashed not in valid_keys:
        raise HTTPException(status_code=403, detail="Wrong API key")
    key_data = valid_keys[hashed]
    logger.info(f"Request from: {key_data.get('name', 'unknown')}")
    return key_data


# ── STEP 5: LOAD ONNX MODEL ──
ort_session   = None
model_cfg     = {}
IMG_SIZE      = 128
EMB_SIZE      = 128
IMG_MEAN      = [0.485, 0.456, 0.406]
IMG_STD       = [0.229, 0.224, 0.225]
MODEL_IS_READY = False


def load_onnx_model():
    global ort_session, model_cfg, IMG_SIZE, EMB_SIZE
    global IMG_MEAN, IMG_STD, MODEL_IS_READY

    config_file = os.path.join(MODELS_DIR, "model_config.json")
    onnx_file   = os.path.join(MODELS_DIR, "muzzle_encoder.onnx")

    if not os.path.exists(config_file):
        logger.error(f"Config missing: {config_file}")
        return

    if not os.path.exists(onnx_file):
        logger.error(f"ONNX missing: {onnx_file}")
        return

    try:
        with open(config_file) as f:
            model_cfg = json.load(f)

        IMG_SIZE = model_cfg.get("image_size", 128)
        EMB_SIZE = model_cfg.get("embedding_size", 128)
        IMG_MEAN = model_cfg.get("normalize_mean", [0.485, 0.456, 0.406])
        IMG_STD  = model_cfg.get("normalize_std",  [0.229, 0.224, 0.225])

        ort_session    = ort.InferenceSession(
            onnx_file,
            providers=["CPUExecutionProvider"]
        )
        MODEL_IS_READY = True
        logger.info(f"MODEL LOADED OK: size={IMG_SIZE} embedding={EMB_SIZE}")

    except Exception as load_err:
        logger.error(f"Model load failed: {load_err}")
        MODEL_IS_READY = False


if all_files_ok:
    load_onnx_model()
else:
    logger.error("Skipping model load because files missing")


# ── STEP 6: IMAGE PROCESSING FUNCTIONS ──
def convert_image_to_array(image_bytes):
    img     = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img     = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr     = np.array(img, dtype=np.float32) / 255.0
    mean_np = np.array(IMG_MEAN, dtype=np.float32)
    std_np  = np.array(IMG_STD,  dtype=np.float32)
    arr     = (arr - mean_np) / std_np
    arr     = arr.transpose(2, 0, 1)
    arr     = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


def get_embedding_from_image(image_bytes):
    if not MODEL_IS_READY:
        raise RuntimeError("Model not ready yet")
    if len(image_bytes) == 0:
        raise ValueError("Image is empty")

    input_arr   = convert_image_to_array(image_bytes)
    output      = ort_session.run(
        ["embedding"],
        {"muzzle_image": input_arr}
    )
    raw_emb     = output[0][0]
    norm_val    = np.linalg.norm(raw_emb)
    if norm_val > 0:
        normalized = raw_emb / norm_val
    else:
        normalized = raw_emb

    return normalized.tolist()


def compute_similarity(emb1, emb2):
    v1  = np.array(emb1, dtype=np.float32)
    v2  = np.array(emb2, dtype=np.float32)
    dot = np.dot(v1, v2)
    return float(np.clip(dot, 0.0, 1.0))


def get_status_from_confidence(confidence_percent):
    if confidence_percent > 90:
        return {
            "status": "CONFIRMED_MATCH",
            "color":  "GREEN",
            "action": "Identity confirmed"
        }
    elif confidence_percent >= 85:
        return {
            "status": "WARNING",
            "color":  "AMBER",
            "action": "Manual verification recommended"
        }
    else:
        return {
            "status": "NO_MATCH",
            "color":  "RED",
            "action": "Re-scan or register as new animal"
        }


# ── STEP 7: API ENDPOINTS ──

@app.get("/")
def home():
    return {
        "service":     "Livestock Guardian Biometric API",
        "version":     "1.0.0",
        "model_ready": MODEL_IS_READY,
        "hf_username": HF_USERNAME,
        "hf_repo":     HF_REPO,
        "docs":        "/docs",
        "time":        datetime.now().isoformat()
    }


@app.get("/health")
def health():
    return {
        "status":         "ok" if MODEL_IS_READY else "model_not_loaded",
        "model_ready":    MODEL_IS_READY,
        "image_size":     IMG_SIZE,
        "embedding_size": EMB_SIZE,
        "time":           datetime.now().isoformat()
    }


@app.get("/debug")
def debug_info():
    files_info = {}
    for fname in FILES_NEEDED:
        fpath = os.path.join(MODELS_DIR, fname)
        exists = os.path.exists(fpath)
        files_info[fname] = {
            "exists":   exists,
            "size_mb":  round(os.path.getsize(fpath) / 1e6, 3) if exists else 0,
            "hf_url":   make_url(fname) if HF_USERNAME else "HF_USERNAME not set"
        }
    return {
        "hf_username":     HF_USERNAME,
        "hf_repo":         HF_REPO,
        "hf_token_set":    bool(HF_TOKEN),
        "model_is_ready":  MODEL_IS_READY,
        "all_files_ready": all_files_ok,
        "files":           files_info
    }


@app.post("/biometric/generate-embedding")
async def generate_embedding(
    file:     UploadFile = File(...),
    key_info: dict       = Depends(check_api_key)
):
    try:
        raw_bytes = await file.read()
        embedding = get_embedding_from_image(raw_bytes)
        return {
            "success":        True,
            "embedding":      embedding,
            "embedding_size": len(embedding),
            "time":           datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/biometric/register-muzzle")
async def register_muzzle(
    file:     UploadFile = File(...),
    key_info: dict       = Depends(check_api_key)
):
    try:
        raw_bytes = await file.read()
        embedding = get_embedding_from_image(raw_bytes)
        return {
            "success":        True,
            "embedding":      embedding,
            "embedding_size": len(embedding),
            "time":           datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/biometric/match-muzzle")
async def match_muzzle(
    file:         UploadFile = File(...),
    records_json: str        = Form(...),
    key_info:     dict       = Depends(check_api_key)
):
    try:
        raw_bytes       = await file.read()
        query_embedding = get_embedding_from_image(raw_bytes)

        records = json.loads(records_json)
        if not records:
            return {
                "match_found": False,
                "message":     "No records in database",
                "embedding":   query_embedding,
                "time":        datetime.now().isoformat()
            }

        best_score = 0.0
        best_id    = None

        for record in records:
            if "livestock_id" not in record:
                continue
            if "embedding" not in record:
                continue
            score = compute_similarity(query_embedding, record["embedding"])
            conf  = round(score * 100, 2)
            if conf > best_score:
                best_score = conf
                best_id    = record["livestock_id"]

        status_info = get_status_from_confidence(best_score)

        return {
            "match_found":    status_info["status"] != "NO_MATCH",
            "livestock_id":   best_id,
            "confidence":     best_score,
            "status":         status_info["status"],
            "color":          status_info["color"],
            "action":         status_info["action"],
            "total_compared": len(records),
            "embedding":      query_embedding,
            "time":           datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


@app.post("/biometric/check-duplicate")
async def check_duplicate(
    file:         UploadFile = File(...),
    records_json: str        = Form(...),
    key_info:     dict       = Depends(check_api_key)
):
    try:
        raw_bytes       = await file.read()
        query_embedding = get_embedding_from_image(raw_bytes)

        records = json.loads(records_json)
        if not records:
            return {
                "is_duplicate": False,
                "message":      "No records yet",
                "embedding":    query_embedding,
                "time":         datetime.now().isoformat()
            }

        highest_conf = 0.0

        for record in records:
            if "livestock_id" not in record:
                continue
            if "embedding" not in record:
                continue
            score = compute_similarity(query_embedding, record["embedding"])
            conf  = round(score * 100, 2)
            if conf > highest_conf:
                highest_conf = conf
            if conf > 90:
                return {
                    "is_duplicate":          True,
                    "duplicate_livestock_id": record["livestock_id"],
                    "confidence":            conf,
                    "embedding":             query_embedding,
                    "time":                  datetime.now().isoformat()
                }

        return {
            "is_duplicate":       False,
            "highest_confidence": highest_conf,
            "message":            "Safe to register",
            "embedding":          query_embedding,
            "time":               datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
