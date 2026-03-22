import logging
import os

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "sergeyzh/rubert-mini-frida")
OUTPUT_PATH = os.getenv("ONNX_MODEL_PATH", "/models/onnx")


def main():
    logger.info("Converting %s to ONNX → %s", MODEL_NAME, OUTPUT_PATH)

    model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    logger.info("Done. Model saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
