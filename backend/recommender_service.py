from typing import List, Dict, Optional
import logging

# Import the real recommender from src/
from src.recommender.recommendation_pipeline import recommend_by_mood, recommend_similar_by_name, recommend_similar_song
from src.recommender.similarity_engine import _get_songs_df

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Mapping from Hume AI emotion names to mood strings used by the
# src/recommender system (happy, sad, energetic, calm, angry, romantic, mellow)
# --------------------------------------------------------------------
EMOTION_TO_MOOD: Dict[str, str] = {
    # Direct mappings
    "joy": "happy",
    "happiness": "happy",
    "amusement": "happy",
    "excitement": "energetic",
    "ecstasy": "energetic",
    "interest": "energetic",
    "surprise": "energetic",
    "sadness": "sad",
    "grief": "sad",
    "disappointment": "sad",
    "distress": "sad",
    "anger": "angry",
    "annoyance": "angry",
    "contempt": "angry",
    "disgust": "angry",
    "calmness": "calm",
    "contentment": "calm",
    "relief": "calm",
    "satisfaction": "calm",
    "serenity": "calm",
    "concentration": "focus",  # Studying/working - needs focus music
    "boredom": "mellow",
    "tiredness": "mellow",
    "contemplation": "mellow",
    "nostalgia": "nostalgic",  # Nostalgic mood - prefers older songs
    "love": "romantic",
    "desire": "romantic",
    "admiration": "romantic",
    "adoration": "romantic",
    "romance": "romantic",
    # Anxiety-like emotions → suggest calming music
    "anxiety": "calm",
    "fear": "calm",
    "horror": "calm",
    "nervousness": "calm",
}

# Default mood previously was "calm". We now avoid defaulting to a mood for unknown emotion.
UNKNOWN_MOOD = "unknown"


def _map_emotion_to_mood(emotion: str) -> str:
    """
    Map a Hume AI emotion name to a mood string understood by the recommender.
    Returns UNKNOWN_MOOD if no mapping is found.
    """
    key = (emotion or "").lower().strip()
    return EMOTION_TO_MOOD.get(key, UNKNOWN_MOOD)


def _dedup_records(records: List[Dict]) -> List[Dict]:
    """
    Remove duplicate recommendations (same track_name + track_artist).
    Keeps first occurrence.
    """
    seen = set()
    out: List[Dict] = []
    for rec in records:
        key = (rec.get("track_name"), rec.get("track_artist"))
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def get_recommendations(top_emotion: str, all_emotions: List[Dict] = None, limit: int = 10) -> List[Dict]:
    """
    Get song recommendations based on detected emotions using the real
    src/recommender system.
    """
    mood = _map_emotion_to_mood(top_emotion)
    logger.info(f"Mapped emotion '{top_emotion}' -> mood '{mood}'")

    # ✅ NEW: If we can't map the emotion, don't default to calm.
    # Return empty list so the UI can ask a follow-up instead of showing random calm music.
    if mood == UNKNOWN_MOOD:
        logger.info(f"Unknown emotion '{top_emotion}'. Returning no recommendations.")
        return []

    try:
        df = recommend_by_mood(mood, n=limit)
        records = df.to_dict(orient="records")

        # Clean up any NaN/inf values that can't be serialized
        for rec in records:
            for k, v in list(rec.items()):
                if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
                    rec[k] = None

        # ✅ NEW: Deduplicate + enforce limit after dedup
        records = _dedup_records(records)
        return records[:limit]

    except Exception as e:
        logger.error(f"Recommender failed: {e}", exc_info=True)
        return []


def get_supported_moods() -> List[str]:
    """
    Get list of moods supported by the recommender.
    """
    return ["happy", "sad", "energetic", "calm", "angry", "romantic", "mellow"]


def get_similar_songs_by_name(
    song_name: str,
    limit: int = 10,
    preset: Optional[str] = None,
    use_genre_boost: bool = True,
    use_artist_diversity: bool = True,
) -> List[Dict]:
    """
    Find songs similar to a given song by searching by name.
    Uses audio feature similarity (danceability, energy, valence, tempo, etc.)
    """
    try:
        df = recommend_similar_by_name(
            song_name,
            n=limit,
            preset=preset,
            use_genre_boost=use_genre_boost,
            use_artist_diversity=use_artist_diversity,
        )

        if df.empty:
            logger.warning(f"No songs found matching: {song_name}")
            return []

        records = df.to_dict(orient="records")

        # Clean up any NaN/inf values
        for rec in records:
            for k, v in list(rec.items()):
                if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
                    rec[k] = None

        # ✅ NEW: Deduplicate + enforce limit after dedup
        records = _dedup_records(records)
        return records[:limit]

    except Exception as e:
        logger.error(f"Similar songs search failed: {e}", exc_info=True)
        return []


def get_song_by_track_id(track_id: str) -> Optional[Dict]:
    """
    Look up a song by its Spotify track ID.
    """
    try:
        songs = _get_songs_df()
        matches = songs[songs["track_id"] == track_id]

        if matches.empty:
            logger.warning(f"No song found with track_id: {track_id}")
            return None

        song = matches.iloc[0].to_dict()

        # Clean up NaN values
        for k, v in list(song.items()):
            if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
                song[k] = None

        return song

    except Exception as e:
        logger.error(f"Track ID lookup failed: {e}", exc_info=True)
        return None


def get_similar_songs_by_track_id(
    track_id: str,
    limit: int = 10,
    preset: Optional[str] = None,
    use_genre_boost: bool = True,
    use_artist_diversity: bool = True,
) -> List[Dict]:
    """
    Find songs similar to a given Spotify track ID.
    Uses audio feature similarity (danceability, energy, valence, tempo, etc.)
    """
    try:
        df = recommend_similar_song(
            track_id,
            n=limit,
            preset=preset,
            use_genre_boost=use_genre_boost,
            use_artist_diversity=use_artist_diversity,
        )

        if df.empty:
            logger.warning(f"No similar songs found for track_id: {track_id}")
            return []

        records = df.to_dict(orient="records")

        # Clean up any NaN/inf values
        for rec in records:
            for k, v in list(rec.items()):
                if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
                    rec[k] = None

        # ✅ NEW: Deduplicate + enforce limit after dedup
        records = _dedup_records(records)
        return records[:limit]

    except KeyError as e:
        logger.warning(f"Track ID not found: {e}")
        return []
    except Exception as e:
        logger.error(f"Similar songs by track_id failed: {e}", exc_info=True)
        return []