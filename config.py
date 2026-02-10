"""
Configuration for Urgency Keywords Test Service (No Redis)
"""
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for Urgency Keywords Test Service."""

    # Service Configuration
    SERVICE_NAME = "urgency_keywords_test"
    SERVICE_VERSION = "1.0.0"
    HOST_ADDR = os.getenv('HOST_ADDR', '0.0.0.0')
    HOST_PORT = int(os.getenv('HOST_PORT', 8000))

    # Model Configuration
    MODEL_DIR = os.getenv('MODEL_DIR', './models/tf_bert_urgency_model')
    URGENCY_THRESHOLD = float(os.getenv('URGENCY_THRESHOLD', 0.4))
    SCALER_METHOD = os.getenv('SCALER_METHOD', 'power')

    # Processing Configuration
    MAX_EMAIL_LENGTH = int(os.getenv('MAX_EMAIL_LENGTH', 10000))

    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    @classmethod
    def validate(cls):
        """Validate configuration values."""
        errors = []

        if not (1 <= cls.HOST_PORT <= 65535):
            errors.append(f"Invalid port number: {cls.HOST_PORT}")

        if cls.URGENCY_THRESHOLD < 0 or cls.URGENCY_THRESHOLD > 1:
            errors.append(f"Invalid urgency threshold: {cls.URGENCY_THRESHOLD}")

        if errors:
            for error in errors:
                logger.error(f"[Config Error]: {error}")
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

        logger.info(f"[Config]: Configuration validated successfully")
        logger.info(f"[Config]: Service={cls.SERVICE_NAME}, Port={cls.HOST_PORT}")
        logger.info(f"[Config]: Model={cls.MODEL_DIR}, Threshold={cls.URGENCY_THRESHOLD}")
        return True
