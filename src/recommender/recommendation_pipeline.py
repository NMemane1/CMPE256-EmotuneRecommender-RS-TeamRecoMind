from __future__ import annotations
import pandas as pd

from src.recommender.similarity_engine import (
    get_mood_recommendations,
    get_similar_songs,
    get_similar_songs_by_name,
)


def recommend_by_mood(mood: str, n: int = 10) -> pd.DataFrame:
    """
    Wrapper used by the UI to get mood-based recommendations.
    """
    return get_mood_recommendations(mood, top_k=n)


def recommend_similar_song(song_id: str, n: int = 10) -> pd.DataFrame:
    """
    Wrapper used by the UI to get songs similar to a given track_id.
    """
    return get_similar_songs(song_id, top_k=n)


def recommend_similar_by_name(song_name: str, n: int = 10) -> pd.DataFrame:
    """
    Find songs similar to a song by name (fuzzy match).
    """
    return get_similar_songs_by_name(song_name, top_k=n)