from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch
import os
import tempfile
from datetime import datetime
import platform
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
# Load Whisper Model ONCE at startup (CPU-only for Railway)
# ----------------------------
print("🚀 Loading Whisper Tiny (CPU mode)...")
MODEL_NAME = "openai/whisper-tiny"

# Force CPU — Railway doesn't support GPU on standard instances
device = "cpu"
print(f"Using device: {device}")

# Disable gradients and set float32 (saves RAM vs float64)
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float32)

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32  # Explicitly use float32
)
model.eval()
model.to(device)

print("✅ Model loaded on CPU")


# ----------------------------
# Root Diagnostic Endpoint
# ----------------------------
@app.get("/")
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
        "endpoints": {
            "transcribe": "/transcribe/ (POST, multipart/form-data with audio file)"
        },
        "ready": True
    }


# ----------------------------
# Transcribe Endpoint
# ----------------------------
@app.post("/transcribe/")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = "en"  # "auto" or ISO code like "es", "fr"
):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1]) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        # Load audio at 16kHz mono (reduce memory by not loading extra channels)
        audio_array, _ = librosa.load(tmp_path, sr=16000, mono=True)

        # Process
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(device)

        generate_kwargs = {"max_new_tokens": 128}
        if language != "auto":
            generate_kwargs["language"] = language
            generate_kwargs["task"] = "transcribe"

        generated_ids = model.generate(input_features, **generate_kwargs)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {
            "text": transcription.strip(),
            "language": language,
            "model": MODEL_NAME
        }

    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                print(f"⚠️ Failed to delete temp file: {e}")
