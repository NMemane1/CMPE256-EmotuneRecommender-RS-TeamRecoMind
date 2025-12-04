from __future__ import annotations

from typing import Any, Dict

import requests

# FastAPI backend from backend/main.py
BACKEND_BASE_URL = "http://localhost:8000"


def recommend_from_text_backend(text: str | None = None, mood: str | None = None) -> Dict[str, Any]:
    """
    Call Arshan's /api/recommend/text endpoint.

    - If mood is given, backend uses it directly.
    - If only text is given, backend uses its heuristic / emotion model.
    Returns the full RecommendationResponse JSON.
    """
    url = f"{BACKEND_BASE_URL}/api/recommend/text"
    payload: Dict[str, Any] = {}
    if text:
        payload["text"] = text
    if mood:
        payload["mood"] = mood

    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def recommend_from_audio_backend(file_obj) -> Dict[str, Any]:
    """
    Call /api/recommend/audio with an uploaded audio file.

    `file_obj` is the Streamlit UploadedFile.
    Returns the full RecommendationResponse JSON.
    """
    url = f"{BACKEND_BASE_URL}/api/recommend/audio"

    file_bytes = file_obj.getvalue()
    filename = file_obj.name or "audio.wav"
    content_type = file_obj.type or "audio/wav"

    files = {
        "file": (filename, file_bytes, content_type)
    }

    resp = requests.post(url, files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()