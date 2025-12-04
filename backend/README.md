# Emotune Recommender Backend

FastAPI backend for the Emotune music recommendation system based on emotional analysis.

## Features

- **Audio Emotion Detection**: Analyze emotions from audio files using Hume AI
- **Text Emotion Detection**: Analyze emotions from text (placeholder for now)
- **Music Recommendations**: Get personalized song recommendations based on detected emotions

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export HUME_API_KEY="your_hume_api_key_here"
```

3. Run the server:
```bash
python main.py
```

Or use uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Root
- `GET /` - API information and available endpoints

### Emotion Detection
- `POST /api/emotion/text` - Analyze emotions from text
  - Request body: `{"text": "your text here"}`
  
- `POST /api/emotion/audio` - Analyze emotions from audio file
  - Form data: `file` (audio file)

### Recommendations
- `POST /api/recommend/text` - Get song recommendations from text
  - Request body: `{"text": "your text here"}`
  
- `POST /api/recommend/audio` - Get song recommendations from audio
  - Form data: `file` (audio file)

## Example Usage

### Using curl

**Text-based recommendation:**
```bash
curl -X POST "http://localhost:8000/api/recommend/text" \

  # Emotune Recommender Backend

  FastAPI backend for the Emotune music recommendation system. This service exposes two primary endpoints:

  - `POST /api/recommend/audio` — upload an audio file, Hume AI prosody model analyzes emotions, and the service returns detected emotions plus song recommendations.
  - `POST /api/recommend/text` — provide `text` or a direct `mood` value to get song recommendations.

  ## Quickstart (backend)

  1. Open a terminal and change into the backend directory:

  ```bash
  cd backend
  ```

  2. Create and activate a virtual environment (macOS/Linux):

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

  3. Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

  4. Configure environment variables:

  - Copy `.env.example` to `.env` and set your Hume API key:

  ```bash
  cp .env.example .env
  # edit .env and set HUME_API_KEY
  ```

  5. Run the server:

  ```bash
  uvicorn main:app --host 127.0.0.1 --port 8000
  ```

  Open the interactive API docs at: `http://127.0.0.1:8000/docs`

  ## Endpoints

  - `POST /api/recommend/audio`
    - Form field: `file` — audio file (common extensions accepted: `.mp3`, `.m4a`, `.wav`, etc.)
    - Returns: detected `emotions`, `top_emotion`, `top_score`, and `recommendations` (demo list for now)

  - `POST /api/recommend/text`
    - JSON body: either `{ "mood": "Joy" }` to directly request recommendations for a mood, or `{ "text": "I feel great" }` to run a simple text heuristic (placeholder).
    - Returns: same response shape as audio endpoint.

  ## Examples

  Text (use mood):

  ```bash
  curl -X POST "http://127.0.0.1:8000/api/recommend/text" \
    -H "Content-Type: application/json" \
    -d '{"mood":"Joy"}'
  ```

  Text (analyze text):

  ```bash
  curl -X POST "http://127.0.0.1:8000/api/recommend/text" \
    -H "Content-Type: application/json" \
    -d '{"text":"I am feeling very sad today"}'
  ```

  Audio (from CLI):

  ```bash
  curl -X POST "http://127.0.0.1:8000/api/recommend/audio" \
    -F "file=@/path/to/audio.m4a;type=audio/mp4"
  ```

  Notes: the OpenAPI docs UI (`/docs`) may not always set a MIME type for uploaded files; the backend will infer audio by filename extension if the content type is missing.

  ## Logs

  - When running in the background with redirection, logs are saved to `/tmp/emotune_uvicorn.log` in the examples I used.

  ## Development notes

  - `emotion_service.py` calls the Hume AI batch API and extracts prosody/emotion predictions.
  - If `HUME_API_KEY` is not set the service will raise an error — ensure `.env` contains a valid key.
  - `recommender_service.py` currently returns hardcoded demo songs; replace `get_recommendations()` with your real recommender logic when ready.

  If you want, I can add a small script to run integration tests against the running server or add a systemd/launchd service file for production deployment.
