from sentence_transformers import SentenceTransformer


class BaseBackend:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()
