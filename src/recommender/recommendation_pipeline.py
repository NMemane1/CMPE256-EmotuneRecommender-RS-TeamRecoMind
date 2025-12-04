from __future__ import annotations
from typing import Optional
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


def recommend_similar_song(
    song_id: str,
    n: int = 10,
    preset: Optional[str] = None,
    use_genre_boost: bool = True,
    use_artist_diversity: bool = True,
) -> pd.DataFrame:
    """
    Wrapper used by the UI to get songs similar to a given track_id.
    
    Args:
        song_id: Spotify track ID
        n: Number of recommendations
        preset: Feature weight preset ("mood", "workout", "chill", "psychedelic")
        use_genre_boost: Boost same-genre tracks
        use_artist_diversity: Limit songs per artist
    """
    return get_similar_songs(
        song_id,
        top_k=n,
        preset=preset,
        use_genre_boost=use_genre_boost,
        use_artist_diversity=use_artist_diversity,
    )


def recommend_similar_by_name(
    song_name: str,
    n: int = 10,
    preset: Optional[str] = None,
    use_genre_boost: bool = True,
    use_artist_diversity: bool = True,
) -> pd.DataFrame:
    """
    Find songs similar to a song by name (fuzzy match).
    
    Args:
        song_name: Song name to search for
        n: Number of recommendations
        preset: Feature weight preset ("mood", "workout", "chill", "psychedelic")
        use_genre_boost: Boost same-genre tracks
        use_artist_diversity: Limit songs per artist
    """
    return get_similar_songs_by_name(
        song_name,
        top_k=n,
        preset=preset,
        use_genre_boost=use_genre_boost,
        use_artist_diversity=use_artist_diversity,
    )