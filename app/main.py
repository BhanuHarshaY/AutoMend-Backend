import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.inference import run_inference
from app.schemas.anomaly import LABEL_NAMES, AnomalyRequest, AnomalyResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
NUM_LABELS = 7


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(device: torch.device):
    """
    Load the Track A RoBERTa classifier and tokenizer.

    If models/config.json exists, load from that directory.
    Otherwise fall back to roberta-base from HuggingFace Hub (dev mode).
    """
    if (MODEL_DIR / "config.json").exists():
        source = str(MODEL_DIR)
        logger.info("Loading model and tokenizer from local directory: %s", source)
    else:
        source = "roberta-base"
        logger.info("models/config.json not found -- loading %s from Hub (dev fallback)", source)

    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=NUM_LABELS)
    model.to(device)
    return model, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    device = _select_device()
    logger.info("Selected device: %s", device)

    t0 = time.perf_counter()
    model, tokenizer = _load_model(device)
    elapsed = time.perf_counter() - t0

    model.eval()
    torch.set_grad_enabled(False)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded in %.2fs | device=%s | parameters=%s",
        elapsed,
        device,
        f"{param_count:,}",
    )

    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.device = device
    app.state.model_loaded = True

    yield

    # Shutdown
    logger.info("Shutting down -- releasing model")
    del app.state.model
    del app.state.tokenizer
    app.state.model_loaded = False


app = FastAPI(title="AutoMend Track A Inference API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": getattr(app.state, "model_loaded", False),
        "device": str(getattr(app.state, "device", "unknown")),
    }


@app.post("/predict_anomaly", response_model=AnomalyResponse)
def predict_anomaly(request: Request, body: AnomalyRequest):
    if not getattr(request.app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        class_id, confidence_score = run_inference(
            request.app.state.model,
            request.app.state.tokenizer,
            body.sequence_ids,
            request.app.state.device,
        )
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail="Inference failed")

    return AnomalyResponse(
        class_id=class_id,
        confidence_score=confidence_score,
        label=LABEL_NAMES[class_id],
    )
