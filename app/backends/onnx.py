import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class OnnxBackend:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.session = ort.InferenceSession(
            f"{model_path}/model.onnx",
            providers=["CPUExecutionProvider"],
        )

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        inputs = self.tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )
        outputs = self.session.run(None, dict(inputs))
        embeddings = self._mean_pooling(outputs[0], inputs["attention_mask"])
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.tolist()

    def encode(self, text: str) -> list[float]:
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=512,
        )
        outputs = self.session.run(None, dict(inputs))
        embedding = self._mean_pooling(outputs[0], inputs["attention_mask"])
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        return embedding[0].tolist()

    def _mean_pooling(
        self, token_embeddings: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        mask = attention_mask[:, :, None].astype(np.float32)
        return (token_embeddings * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
