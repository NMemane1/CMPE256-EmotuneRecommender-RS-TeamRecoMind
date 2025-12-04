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
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling so happy today!"}'
```

**Audio-based recommendation:**
```bash
curl -X POST "http://localhost:8000/api/recommend/audio" \
  -F "file=@/path/to/your/audio.m4a"
```

### Using Python requests

```python
import requests

# Text recommendation
response = requests.post(
    "http://localhost:8000/api/recommend/text",
    json={"text": "I am feeling so happy today!"}
)
print(response.json())

# Audio recommendation
with open("audio.m4a", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/recommend/audio",
        files={"file": f}
    )
print(response.json())
```

## Response Format

```json
{
  "emotions": [
    {"name": "Joy", "score": 0.75},
    {"name": "Sadness", "score": 0.15}
  ],
  "top_emotion": "Joy",
  "top_score": 0.75,
  "recommendations": [
    {
      "title": "Happy",
      "artist": "Pharrell Williams",
      "genre": "Pop",
      "year": 2013
    }
  ]
}
```

## Development

The backend is structured as follows:
- `main.py` - FastAPI application and endpoints
- `emotion_service.py` - Emotion detection using Hume AI
- `recommender_service.py` - Music recommendation logic
- `requirements.txt` - Python dependencies

## Notes

- The text emotion detection is currently a placeholder
- Song recommendations are hardcoded for demonstration
- The system will be integrated with a full recommendation engine in future updates
