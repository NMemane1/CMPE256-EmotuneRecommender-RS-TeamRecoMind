import os
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from emotion_service import EmotionDetectionService
from recommender_service import get_recommendations

app = FastAPI(
    title="Emotune Recommender API",
    description="Music recommendation system based on emotional analysis of text and audio",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize emotion detection service
emotion_service = EmotionDetectionService()


class TextRequest(BaseModel):
    text: Optional[str] = None
    # If provided, 'mood' will be used directly to fetch recommendations
    mood: Optional[str] = None


class RecommendationResponse(BaseModel):
    emotions: List[dict]
    top_emotion: str
    top_score: float
    recommendations: List[dict]


@app.get("/")
async def root():
    return {
        "message": "Emotune Recommender API",
        "endpoints": {
            "recommend_from_audio": "POST /api/recommend/audio",
            "recommend_from_text": "POST /api/recommend/text"
        }
    }


@app.post("/api/recommend/audio", response_model=RecommendationResponse)
async def recommend_from_audio(file: UploadFile = File(...)):
    """
    Upload an audio file to detect emotions using Hume AI prosody analysis,
    then get personalized song recommendations based on detected emotions.
    
    Flow:
    1. User uploads audio file
    2. Hume AI analyzes emotions/prosody from the audio
    3. Emotions are passed to the recommender system
    4. Returns detected emotions and recommended songs
    """
    # Validate file type: accept if content_type indicates audio/video, or infer from filename extension
    allowed_exts = {'.mp3', '.m4a', '.wav', '.flac', '.aac', '.ogg', '.opus', '.wma', '.m4b', '.mp4'}
    def _looks_like_audio(upload_file: UploadFile) -> bool:
        if upload_file.content_type and upload_file.content_type.startswith(('audio/', 'video/')):
            return True
        # Fallback: infer from filename extension
        if upload_file.filename:
            _, ext = os.path.splitext(upload_file.filename.lower())
            if ext in allowed_exts:
                return True
        return False

    if not _looks_like_audio(file):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an audio or video file. If using the OpenAPI docs, set the file's MIME to an audio type."
        )
    
    # Save uploaded file temporarily
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Step 1: Analyze emotions using Hume AI
        emotions = await emotion_service.analyze_audio(tmp_path)
        
        top_emotion = emotions[0]["name"]
        top_score = emotions[0]["score"]
        
        # Step 2: Pass emotions to recommender system and get song recommendations
        # TODO: Replace with actual recommender system integration
        recommendations = get_recommendations(top_emotion, emotions)
        
        return RecommendationResponse(
            emotions=emotions,
            top_emotion=top_emotion,
            top_score=top_score,
            recommendations=recommendations
        )
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/recommend/text", response_model=RecommendationResponse)
async def recommend_from_text(request: TextRequest):
    """
    Analyze text to detect emotions, then get personalized song recommendations.
    
    Flow:
    1. User submits text
    2. Text is analyzed for emotions (placeholder for now)
    3. Emotions are passed to the recommender system
    4. Returns detected emotions and recommended songs
    """
    # If user provided a mood, use it directly for recommendations
    if request.mood:
        top_emotion = request.mood
        # We return a minimal emotions list derived from the provided mood
        emotions = [{"name": top_emotion, "score": 1.0}]
        recommendations = get_recommendations(top_emotion, emotions)

        return RecommendationResponse(
            emotions=emotions,
            top_emotion=top_emotion,
            top_score=1.0,
            recommendations=recommendations
        )

    # TODO: Integrate actual text emotion analysis (e.g., using transformers)
    # For now, analyze 'text' if provided, otherwise return an error
    if not request.text:
        raise HTTPException(status_code=400, detail="Either 'text' or 'mood' must be provided")

    # Placeholder simple heuristic: look for common mood words in text
    txt = request.text.lower()
    if any(word in txt for word in ["happy", "joy", "glad", "excited", "elated"]):
        emotions = [{"name": "Joy", "score": 0.9}]
    elif any(word in txt for word in ["sad", "depressed", "unhappy", "down"]):
        emotions = [{"name": "Sadness", "score": 0.9}]
    elif any(word in txt for word in ["angry", "mad", "furious", "annoyed"]):
        emotions = [{"name": "Anger", "score": 0.9}]
    else:
        emotions = [{"name": "Neutral", "score": 0.6}]

    top_emotion = emotions[0]["name"]
    top_score = emotions[0]["score"]
    recommendations = get_recommendations(top_emotion, emotions)

    return RecommendationResponse(
        emotions=emotions,
        top_emotion=top_emotion,
        top_score=top_score,
        recommendations=recommendations
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
