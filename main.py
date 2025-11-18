from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch
import os
import tempfile
from datetime import datetime
import platform
import asyncio
from functools import partial
import transformers

# ----------------------------
# Initialize FastAPI App
# ----------------------------
app = FastAPI(
    title="Whisper Quiz Transcriber",
    description="Transcribe short audio clips for language quiz scoring",
    version="1.0"
)

# ----------------------------
# Add CORS Middleware (REQUIRED for browser frontend)
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Configuration
# ----------------------------
MODEL_NAME = "openai/whisper-tiny"
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB
MAX_DURATION = 30.0               # seconds

# ----------------------------
# Load Whisper Model ONCE at startup (CPU-only)
# ----------------------------
print("🚀 Loading Whisper Tiny (CPU mode)...")
device = "cpu"
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float32)

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)
model.eval()
model.to(device)
print("✅ Model loaded on CPU")


# ----------------------------
# Synchronous Transcription Function (runs in thread)
# ----------------------------
def _transcribe_sync(audio_bytes: bytes, language: str):
    """CPU-heavy work — runs in thread pool"""
    # Save to temp file
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        # Load audio
        audio_array, _ = librosa.load(tmp.name, sr=16000, mono=True)

    # Check duration
    duration = librosa.get_duration(y=audio_array, sr=16000)
    if duration > MAX_DURATION:
        raise ValueError(f"Audio too long: {duration:.1f}s (max {MAX_DURATION}s)")

    # Preprocess
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)

    # Generate
    generate_kwargs = {"max_new_tokens": 128, "task": "transcribe"}
    if language != "auto":
        generate_kwargs["language"] = language

    with torch.no_grad():
        generated_ids = model.generate(input_features, **generate_kwargs)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {
        "text": transcription.strip(),
        "language": language,
        "duration_sec": round(duration, 2),
        "model": MODEL_NAME
    }


# ----------------------------
# Root Diagnostic Endpoint
# ----------------------------
@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {
        "status": "✅ OK",
        "app": "Whisper Quiz Transcriber",
        "version": "1.0",
        "model_loaded": MODEL_NAME,
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "server_time_utc": datetime.utcnow().isoformat() + "Z",
        "ready": True
    }


# ----------------------------
# Transcribe Endpoint (async wrapper)
# ----------------------------
@app.post("/transcribe/", status_code=status.HTTP_200_OK)
async def transcribe(
    audio: UploadFile = File(...),
    language: str = "auto"  # Default to auto-detect
):
    # Validate file presence
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate content type (basic check)
    if audio.content_type and not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail="File must be an audio type")

    # Read content
    content = await audio.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_SIZE // 1024 // 1024} MB)")

    # Offload to thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,  # Uses default thread pool
            partial(_transcribe_sync, content, language)
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
