"""
Urgency Keywords Test Service (No Redis)

FastAPI microservice for testing urgency detection directly.
User provides subject and body directly, no Redis dependency.
"""
import logging
import math
import uvicorn
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from config import Config
from urgency_model import get_urgency_detector

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Urgency Keywords Test Service",
    description="Test urgency detection directly without Redis",
    version=Config.SERVICE_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmailInput(BaseModel):
    """Input model for urgency detection."""
    subject: Optional[str] = Field(default="", description="Email subject")
    body: Optional[str] = Field(default="", description="Email body content")


class UrgencyResponse(BaseModel):
    """Response model for urgency detection."""
    signal: str = Field(..., description="Signal name (always 'urgency_keywords_present')")
    value: float = Field(..., description="Urgency probability score (0.0 to 1.0)")
    input_text: str = Field(..., description="Combined input text used for prediction")
    input_length: int = Field(..., description="Length of input text")


class HealthResponse(BaseModel):
    """Response model for health check."""
    service: str
    version: str
    status: str
    model: str
    model_type: str
    timestamp: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service health status including model availability.
    """
    model_status = "unknown"
    model_type = "unknown"

    try:
        detector = get_urgency_detector()
        if detector.model is not None:
            model_status = "healthy"
            model_type = detector.model_type
        else:
            model_status = "not_loaded"
    except Exception as e:
        logger.error(f"[HealthCheck]: Model health check failed: {e}")
        model_status = "unhealthy"

    overall_status = "healthy" if model_status == "healthy" else "degraded"

    return HealthResponse(
        service=Config.SERVICE_NAME,
        version=Config.SERVICE_VERSION,
        status=overall_status,
        model=model_status,
        model_type=model_type,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=UrgencyResponse)
async def predict_urgency(email: EmailInput) -> UrgencyResponse:
    """
    Detect urgency keywords in email content.

    Directly accepts subject and body, combines them, and runs prediction.

    Args:
        email: EmailInput with subject and body

    Returns:
        UrgencyResponse with signal name and urgency probability score
    """
    start_time = datetime.now()

    subject = email.subject or ""
    body = email.body or ""

    # Combine subject and body (same as redis_handler does)
    combined_text = f"{subject}\n\n{body}".strip()

    logger.info(f"[Predict]: Request received - subject_len={len(subject)}, body_len={len(body)}")

    if not combined_text:
        logger.warning("[Predict]: Empty input, returning 0.0")
        return UrgencyResponse(
            signal="urgency_keywords_present",
            value=0.0,
            input_text="",
            input_length=0
        )

    # Truncate if too long
    if len(combined_text) > Config.MAX_EMAIL_LENGTH:
        logger.info(f"[Predict]: Truncating from {len(combined_text)} to {Config.MAX_EMAIL_LENGTH}")
        combined_text = combined_text[:Config.MAX_EMAIL_LENGTH]

    try:
        # Get urgency detector and predict
        detector = get_urgency_detector()
        result = detector.predict(combined_text)

        # Handle NaN/Inf values from model
        urgency_value = result.get('value', 0.0)
        if urgency_value is None or math.isnan(urgency_value) or math.isinf(urgency_value):
            logger.warning(f"[Predict]: Model returned invalid value ({urgency_value}), defaulting to 0.0")
            urgency_value = 0.0

        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.info(f"[Predict]: Completed - value={urgency_value:.4f}, time={processing_time}ms")

        return UrgencyResponse(
            signal=result.get('signal', 'urgency_keywords_present'),
            value=urgency_value,
            input_text=combined_text[:200] + "..." if len(combined_text) > 200 else combined_text,
            input_length=len(combined_text)
        )

    except Exception as e:
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.error(f"[Predict]: Error - {str(e)}, time={processing_time}ms", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """
    Initialize service on startup.

    Validates configuration and loads the ML model.
    """
    logger.info(f"[Startup]: Initializing {Config.SERVICE_NAME} v{Config.SERVICE_VERSION}")

    try:
        # Validate configuration
        Config.validate()

        # Initialize ML model (singleton pattern loads on first access)
        detector = get_urgency_detector()
        logger.info(f"[Startup]: Urgency detector initialized - model_type={detector.model_type}")

        logger.info(f"[Startup]: Service started successfully on port {Config.HOST_PORT}")

    except Exception as e:
        logger.error(f"[Startup]: Failed to initialize service: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("[Shutdown]: Service shutdown complete")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=Config.HOST_ADDR,
        port=Config.HOST_PORT,
        log_level=Config.LOG_LEVEL.lower()
    )
