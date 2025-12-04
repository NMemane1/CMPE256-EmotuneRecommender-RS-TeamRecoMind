import os
import time
import json
from typing import Dict, List, Tuple
import asyncio
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.hume.ai/v0/batch/jobs"


class EmotionDetectionService:
    """
    Service for detecting emotions from audio using Hume AI API
    """
    
    def __init__(self):
        self.api_key = self._get_api_key()
        self.logger = logging.getLogger("emotion_service")
        if not logging.getLogger().handlers:
            # Basic logging configuration when not configured by the app
            logging.basicConfig(level=logging.INFO)
    
    def _get_api_key(self) -> str:
        """Get Hume API key from environment variable. Returns None if not set."""
        api_key = os.getenv('HUME_API_KEY')
        if not api_key:
            # Explicitly return None when the key isn't configured
            return None
        return api_key
    
    def _start_job_from_local_file(self, audio_path: str) -> str:
        """
        Start a batch inference job from a local file
        """
        url = BASE_URL
        headers = {
            "X-Hume-Api-Key": self.api_key,
        }

        job_config = {
            "models": {
                "prosody": {}
            }
        }

        with open(audio_path, "rb") as f:
            files = {
                "file": (os.path.basename(audio_path), f, "application/octet-stream")
            }
            data = {
                "json": json.dumps(job_config)
            }

            resp = requests.post(url, headers=headers, data=data, files=files)

        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise RuntimeError(f"Start job failed: {resp.status_code} - {body}")

        body = resp.json()
        job_id = body.get("job_id")
        if not job_id:
            raise RuntimeError(f"No job_id in response: {body}")
        return job_id

    def _get_job_details(self, job_id: str) -> dict:
        """Get job details (state, status)"""
        url = f"{BASE_URL}/{job_id}"
        headers = {
            "X-Hume-Api-Key": self.api_key,
            "accept": "application/json; charset=utf-8",
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def _get_job_predictions(self, job_id: str) -> list:
        """Get job predictions"""
        url = f"{BASE_URL}/{job_id}/predictions"
        headers = {
            "X-Hume-Api-Key": self.api_key,
            "accept": "application/json; charset=utf-8",
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def _extract_emotions_from_prosody(self, predictions: list) -> List[Tuple[str, float]]:
        """
        Extract emotions from prosody predictions
        """
        all_emotions: Dict[str, float] = {}

        for source_entry in predictions:
            results = source_entry.get("results", {})
            preds = results.get("predictions", [])
            for pred in preds:
                models = pred.get("models", {})
                prosody = models.get("prosody")
                if not prosody:
                    continue

                grouped = prosody.get("grouped_predictions", [])
                for group in grouped:
                    for p in group.get("predictions", []):
                        for emo in p.get("emotions", []):
                            name = emo.get("name")
                            score = emo.get("score", 0.0)
                            if not name:
                                continue
                            if name not in all_emotions or score > all_emotions[name]:
                                all_emotions[name] = score

        return sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)

    async def analyze_audio(self, audio_path: str, max_wait_time: int = 120) -> List[dict]:
        """
        Analyze audio file and return emotion predictions
        
        Args:
            audio_path: Path to the audio file
            max_wait_time: Maximum time to wait for job completion (seconds)
            
        Returns:
            List of emotion dictionaries with 'name' and 'score' keys
        """
        # If API key is not configured, log and raise an error (do not return hardcoded emotions)
        if not self.api_key:
            self.logger.error("HUME API key not configured; set HUME_API_KEY in environment")
            raise RuntimeError("HUME API key not configured")

        # Start the job
        job_id = self._start_job_from_local_file(audio_path)

        # Poll for completion
        max_attempts = max_wait_time // 3
        for attempt in range(max_attempts):
            job = self._get_job_details(job_id)
            state = job.get("state", {})
            status = state.get("status")

            if status == "COMPLETED":
                break
            if status == "FAILED":
                raise RuntimeError(f"Job failed: {json.dumps(job, indent=2)}")

            # Wait before next check
            await asyncio.sleep(3)
        else:
            raise RuntimeError("Timed out waiting for job to complete")

        # Get predictions
        predictions = self._get_job_predictions(job_id)
        emotions_tuple = self._extract_emotions_from_prosody(predictions)

        if not emotions_tuple:
            # No prosody detected (e.g., instrumental music, no speech)
            # Return a neutral fallback so the API doesn't crash
            self.logger.warning(
                "No prosody emotion predictions found in Hume response. "
                "This may happen with instrumental music or short/silent audio. "
                "Falling back to neutral emotion."
            )
            return [{"name": "Calmness", "score": 0.5}]

        # Convert to list of dicts
        emotions = [
            {"name": name, "score": score}
            for name, score in emotions_tuple
        ]

        return emotions
