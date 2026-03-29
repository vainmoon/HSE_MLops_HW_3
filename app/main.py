import logging
import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backends import load_backend
from backends.dynamic import DynamicBatchingBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "sergeyzh/rubert-mini-frida")
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "base")
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "/models/onnx")
DYNAMIC_BATCHING = os.getenv("DYNAMIC_BATCHING", "false").lower() == "true"
BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "32"))
BATCH_MAX_WAIT_MS = float(os.getenv("BATCH_MAX_WAIT_MS", "20"))

backend = None
dynamic_backend: DynamicBatchingBackend | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global backend, dynamic_backend

    logger.info("Loading backend=%s", INFERENCE_BACKEND)
    backend = load_backend(
        backend=INFERENCE_BACKEND,
        model_name=MODEL_NAME,
        onnx_model_path=ONNX_MODEL_PATH,
    )
    logger.info("Backend loaded successfully")

    if DYNAMIC_BATCHING:
        dynamic_backend = DynamicBatchingBackend(
            inner_backend=backend,
            max_batch_size=BATCH_MAX_SIZE,
            max_wait_ms=BATCH_MAX_WAIT_MS,
        )
        await dynamic_backend.start()

    yield

    if dynamic_backend:
        await dynamic_backend.stop()
    backend = None


app = FastAPI(title="Embedding Service", lifespan=lifespan)


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": INFERENCE_BACKEND,
        "model": MODEL_NAME,
        "dynamic_batching": DYNAMIC_BATCHING,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    if backend is None:
        raise HTTPException(status_code=503, detail="Backend not loaded")

    start = time.perf_counter()
    embedding = backend.encode(request.text)
    latency_ms = (time.perf_counter() - start) * 1000

    logger.debug("Encoded text of length %d in %.2f ms", len(request.text), latency_ms)

    return EmbedResponse(embedding=embedding)


@app.post("/async_embed", response_model=EmbedResponse)
async def async_embed(request: EmbedRequest):
    if dynamic_backend is None:
        raise HTTPException(
            status_code=503,
            detail="Dynamic batching is not enabled. Set DYNAMIC_BATCHING=true.",
        )

    start = time.perf_counter()
    embedding = await dynamic_backend.encode(request.text)
    latency_ms = (time.perf_counter() - start) * 1000

    logger.debug(
        "Async encoded text of length %d in %.2f ms", len(request.text), latency_ms
    )

    return EmbedResponse(embedding=embedding)
