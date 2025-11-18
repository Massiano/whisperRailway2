from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import torch
import os
import tempfile

# ----------------------------
# Initialize FastAPI App
# ----------------------------
app = FastAPI(
    title="Whisper Quiz Transcriber",
    description="Transcribe short audio clips for language quiz scoring",
    version="1.0"
)

# ----------------------------
# Load Whisper Model ONCE at startup
# ----------------------------
print("Loading Whisper Tiny...")
MODEL_NAME = "openai/whisper-tiny"  # ~180MB RAM, supports 99 languages

# Disable gradient computation for memory savings
torch.set_grad_enabled(False)

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()  # Set to inference mode
print("Model loaded ✅")

# ----------------------------
# Transcribe Endpoint
# ----------------------------
@app.post("/transcribe/")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = "en"  # Optional: force language (e.g., "es", "fr", "de")
):
    """
    Accepts a short audio file (WAV, MP3, etc.), returns transcription.
    """
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Use secure temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1]) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        # Load audio at 16kHz mono
        audio_array, _ = librosa.load(tmp_path, sr=16000)

        # Process for Whisper
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features

        # Force language if provided (improves accuracy)
        forced_decoder_ids = None
        if language != "auto":
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

        # Generate transcription
        generated_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=128  # Prevent long outputs
        )

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {
            "text": transcription.strip(),
            "language": language,
            "model": MODEL_NAME
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
