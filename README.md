# Whisper Quiz Backend

Transcribe short audio clips for language learning apps using Whisper Tiny.

## Features
- Runs `openai/whisper-tiny` (99 languages)
- Optimized for Railway.app (CPU, <512MB RAM)
- Accepts MP3, WAV, etc.
- Language hinting for better accuracy

## Deploy to Railway
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/your-username/whisper-quiz-backend)

## Local Dev
```bash
pip install -r requirements.txt
uvicorn main:app --reload
