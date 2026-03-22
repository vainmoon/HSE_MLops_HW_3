import logging
import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backends import load_backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "sergeyzh/rubert-mini-frida")
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "base")
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "/models/onnx")

backend = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global backend
    logger.info("Loading backend=%s", INFERENCE_BACKEND)
    backend = load_backend(
        backend=INFERENCE_BACKEND,
        model_name=MODEL_NAME,
        onnx_model_path=ONNX_MODEL_PATH,
    )
    logger.info("Backend loaded successfully")
    yield
    backend = None


app = FastAPI(title="Embedding Service", lifespan=lifespan)


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]


@app.get("/health")
def health():
    return {"status": "ok", "backend": INFERENCE_BACKEND, "model": MODEL_NAME}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    if backend is None:
        raise HTTPException(status_code=503, detail="Backend not loaded")

    start = time.perf_counter()
    embedding = backend.encode(request.text)
    latency_ms = (time.perf_counter() - start) * 1000

    logger.debug("Encoded text of length %d in %.2f ms", len(request.text), latency_ms)

    return EmbedResponse(embedding=embedding)
