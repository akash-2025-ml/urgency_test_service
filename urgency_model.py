"""
Urgency Detection Model Wrapper

BERT-based urgency detection model adapted from ML engineer's implementation.
Provides a clean interface for the FastAPI service to use.
"""
import os
import warnings
import logging
import numpy as np

# Suppress TensorFlow warnings before importing
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", message=".*TensorFlow and JAX classes are deprecated.*")
warnings.filterwarnings("ignore")

import tensorflow as tf
from transformers import AutoTokenizer

from config import Config

# Suppress TensorFlow logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class UrgencyDetector:
    """
    BERT-based urgency detection model.

    Uses a fine-tuned BERT model to detect urgency keywords and phrases
    in email content. Returns a probability score between 0.0 and 1.0.
    """

    def __init__(self, model_dir: str = None):
        """
        Initialize the urgency detector.

        Args:
            model_dir: Path to the TensorFlow model directory
        """
        self.model_dir = model_dir or Config.MODEL_DIR
        self.threshold = Config.URGENCY_THRESHOLD
        self.scaler_method = Config.SCALER_METHOD
        self.tokenizer = None
        self.model = None
        self.model_type = None  # 'keras' or 'saved_model'
        self._load_model()

    def _load_model(self):
        """Load tokenizer and model."""
        if not os.path.exists(self.model_dir):
            raise RuntimeError(f"[UrgencyDetector]: Model directory not found at {self.model_dir}")

        logger.info(f"[UrgencyDetector]: Loading model from {self.model_dir}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        logger.info("[UrgencyDetector]: Tokenizer loaded (bert-base-uncased)")

        # Try Keras model first
        try:
            logger.info("[UrgencyDetector]: Attempting tf.keras.models.load_model...")
            self.model = tf.keras.models.load_model(self.model_dir)
            self.model_type = 'keras'
            logger.info("[UrgencyDetector]: Model loaded successfully using Keras")
            return
        except Exception as e:
            logger.debug(f"[UrgencyDetector]: Keras load failed: {e}")

        # Fallback to SavedModel
        try:
            logger.info("[UrgencyDetector]: Attempting tf.saved_model.load...")
            self.model = tf.saved_model.load(self.model_dir)
            self.model_type = 'saved_model'
            logger.info("[UrgencyDetector]: Model loaded successfully using SavedModel")
            return
        except Exception as e:
            logger.error(f"[UrgencyDetector]: SavedModel load failed: {e}")
            raise RuntimeError(f"Failed to load model from {self.model_dir}")

    def _extract_scalar(self, output) -> float:
        """Extract scalar probability from model output."""
        if isinstance(output, dict):
            first = list(output.values())[0]
            if isinstance(first, tf.Tensor):
                return float(first.numpy().flatten()[0])
            return float(np.asarray(first).flatten()[0])
        elif isinstance(output, tf.Tensor):
            return float(output.numpy().flatten()[0])
        else:
            return float(np.asarray(output).flatten()[0])

    def predict(self, text: str) -> dict:
        """
        Predict urgency score for given text.

        Args:
            text: Email content (subject + body combined)

        Returns:
            Dictionary with signal name and probability value:
            {"signal": "urgency_keywords_present", "value": <float 0.0-1.0>}
        """
        try:
            # Tokenize input
            encoding = self.tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="tf",
            )
            input_ids = tf.cast(encoding["input_ids"], tf.int32)
            attention_mask = tf.cast(encoding["attention_mask"], tf.int32)

            # Predict based on model type
            if self.model_type == 'keras':
                preds = self.model.predict((input_ids, attention_mask), verbose=0)
                prob_raw = float(np.asarray(preds).flatten()[0])
            else:
                # SavedModel with signature
                serving_fn = self.model.signatures.get("serving_default")
                if serving_fn is None:
                    # Try first available signature
                    sigs = list(getattr(self.model, "signatures", {}).keys())
                    if sigs:
                        serving_fn = self.model.signatures[sigs[0]]
                    else:
                        raise RuntimeError("No serving signature found in model")

                # Get input names from signature
                try:
                    _, in_sig = serving_fn.structured_input_signature
                    input_names = list(in_sig.keys())
                except Exception:
                    input_names = ["input_ids", "attention_mask"]

                # Build input mapping
                if len(input_names) >= 2:
                    mapping = {input_names[0]: input_ids, input_names[1]: attention_mask}
                else:
                    mapping = {"input_ids": input_ids, "attention_mask": attention_mask}

                out = serving_fn(**mapping)
                prob_raw = self._extract_scalar(out)

            logger.info(f"[UrgencyDetector]: Prediction completed - raw_prob={prob_raw:.4f}")

            return {
                "signal": "urgency_keywords_present",
                "value": prob_raw
            }

        except Exception as e:
            logger.error(f"[UrgencyDetector]: Prediction failed: {e}")
            raise


# Singleton instance
_detector = None


def get_urgency_detector() -> UrgencyDetector:
    """
    Get singleton UrgencyDetector instance.

    The model is loaded once and reused for all predictions
    to avoid repeated loading overhead.

    Returns:
        UrgencyDetector instance
    """
    global _detector
    if _detector is None:
        _detector = UrgencyDetector()
    return _detector
