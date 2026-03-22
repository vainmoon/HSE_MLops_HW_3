from .base import BaseBackend
from .onnx import OnnxBackend


def load_backend(backend: str, model_name: str, onnx_model_path: str):
    if backend == "base":
        return BaseBackend(model_name)
    if backend == "onnx":
        return OnnxBackend(onnx_model_path)
    raise ValueError(f"Unknown backend: {backend!r}. Choose 'base' or 'onnx'.")
