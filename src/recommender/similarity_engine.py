from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender.feature_builder import (
    load_song_data,
    build_feature_matrix,
    FEATURE_WEIGHT_PRESETS,
)


# --------------------------------------------------------------------
# Global in-memory cache for songs and feature matrix
# --------------------------------------------------------------------
_SONGS_DF: pd.DataFrame | None = None
_FEATURE_MATRIX: np.ndarray | None = None
_FEATURE_COLS: list[str] | None = None
_CURRENT_PRESET: str | None = None  # Track which preset is cached


def _get_songs_df() -> pd.DataFrame:
    """Load songs.csv once and cache it."""
    global _SONGS_DF
    if _SONGS_DF is None:
        _SONGS_DF = load_song_data()
    return _SONGS_DF


def _get_feature_matrix(
    preset: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Use build_feature_matrix from feature_builder so whatever you defined
    there (danceability/energy/valence/tempo/etc.) is reused consistently.
    
    Args:
        preset: Optional preset name (e.g., "mood", "workout", "chill", "psychedelic")
        weights: Optional custom weights dict
        
    Note: We don't cache when a preset or weights are provided, since different
    presets produce different matrices.
    """
    global _FEATURE_MATRIX, _FEATURE_COLS, _CURRENT_PRESET

    songs = _get_songs_df()
    
    # If using a preset or custom weights, always rebuild (no caching)
    if preset or weights:
        X, feature_cols = build_feature_matrix(songs, weights=weights, preset=preset)
        return X, feature_cols

    # For default (no preset), use cached version
    if _FEATURE_MATRIX is None or _FEATURE_COLS is None:
        X, feature_cols = build_feature_matrix(songs)
        _FEATURE_MATRIX = X
        _FEATURE_COLS = feature_cols
        _CURRENT_PRESET = "default"

    return _FEATURE_MATRIX, _FEATURE_COLS


# --------------------------------------------------------------------
# Genre boosting helper
# --------------------------------------------------------------------
GENRE_BOOST_FACTOR = 1.15  # Boost same-genre tracks by 15%
SUBGENRE_BOOST_FACTOR = 1.10  # Additional boost for same subgenre


def _apply_genre_boost(
    result: pd.DataFrame,
    ref_genre: Optional[str],
    ref_subgenre: Optional[str],
) -> pd.DataFrame:
    """
    Boost similarity scores for tracks in the same genre/subgenre.
    """
    if ref_genre is None or "playlist_genre" not in result.columns:
        return result
    
    result = result.copy()
    
    # Boost same genre
    same_genre_mask = result["playlist_genre"].str.lower() == ref_genre.lower()
    result.loc[same_genre_mask, "similarity"] *= GENRE_BOOST_FACTOR
    
    # Additional boost for same subgenre
    if ref_subgenre and "playlist_subgenre" in result.columns:
        same_subgenre_mask = result["playlist_subgenre"].str.lower() == ref_subgenre.lower()
        result.loc[same_subgenre_mask, "similarity"] *= SUBGENRE_BOOST_FACTOR
    
    return result


# --------------------------------------------------------------------
# Artist diversity helper
# --------------------------------------------------------------------
MAX_PER_ARTIST = 2  # Max songs from same artist in recommendations


def _apply_artist_diversity(result: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    Limit the number of songs per artist to avoid repetitive recommendations.
    """
    if "track_artist" not in result.columns:
        return result.head(top_k)
    
    result = result.sort_values("similarity", ascending=False)
    
    selected = []
    artist_counts: Dict[str, int] = {}
    
    for _, row in result.iterrows():
        artist = row.get("track_artist", "Unknown")
        if artist_counts.get(artist, 0) < MAX_PER_ARTIST:
            selected.append(row)
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            if len(selected) >= top_k:
                break
    
    return pd.DataFrame(selected)


# --------------------------------------------------------------------
# Mood profiles (Option B – including mellow)
# --------------------------------------------------------------------
# These are target points in (valence, energy) space:
#   valence: 0 (negative) → 1 (positive)
#   energy : 0 (calm)     → 1 (intense)
#
# "mellow" ≈ warm, slightly positive, low-mid energy.
MOOD_TARGETS: Dict[str, Dict[str, float]] = {
    "happy":     {"valence": 0.85, "energy": 0.70},
    "sad":       {"valence": 0.20, "energy": 0.30},
    "energetic": {"valence": 0.75, "energy": 0.90},
    "calm":      {"valence": 0.55, "energy": 0.25},
    "angry":     {"valence": 0.25, "energy": 0.85},
    "romantic":  {"valence": 0.80, "energy": 0.55},
    "mellow":    {"valence": 0.55, "energy": 0.45},
}


def _add_mood_distance(
    df: pd.DataFrame,
    mood: str,
    valence_col: str = "valence",
    energy_col: str = "energy",
) -> pd.DataFrame:
    """
    Compute distance from each track to the mood's target (valence, energy).
    Distance is Euclidean in (valence, energy) space.
    """
    mood_key = mood.lower()
    target = MOOD_TARGETS.get(mood_key)

    # Fallback if someone types a mood we didn't define
    if target is None:
        target = MOOD_TARGETS["happy"]
        mood_key = "happy"

    if valence_col not in df.columns or energy_col not in df.columns:
        raise KeyError(
            f"Expected '{valence_col}' and '{energy_col}' columns in songs dataframe."
        )

    v = df[valence_col].astype(float)
    e = df[energy_col].astype(float)

    dv = v - float(target["valence"])
    de = e - float(target["energy"])

    df = df.copy()
    df["mood_distance"] = np.sqrt(dv**2 + de**2)
    # Turn distance into a similarity-like score in [0, 1]
    df["similarity"] = 1.0 / (1.0 + df["mood_distance"])
    return df


# --------------------------------------------------------------------
# Public API – used by recommendation_pipeline.py
# --------------------------------------------------------------------
def get_mood_recommendations(mood: str, top_k: int = 10) -> pd.DataFrame:
    """
    Return top_k tracks that best match the requested mood based on
    valence & energy, with an explanation string for each track.
    """
    songs = _get_songs_df()
    scored = _add_mood_distance(songs, mood)
    scored = scored.sort_values("mood_distance", ascending=True).head(top_k)

    # Explanations for UI / "Why this song"
    def _explain(row: pd.Series) -> str:
        v = row.get("valence", np.nan)
        e = row.get("energy", np.nan)
        return (
            f"Recommended because it matches the '{mood}' mood "
            f"(valence={v:.2f}, energy={e:.2f})."
        )

    scored = scored.copy()
    scored["explanation"] = scored.apply(_explain, axis=1)

    preferred_cols = [
        "track_id",
        "track_name",
        "track_artist",
        "valence",
        "energy",
        "similarity",
        "explanation",
    ]
    ordered = [c for c in preferred_cols if c in scored.columns] + [
        c for c in scored.columns if c not in preferred_cols
    ]
    return scored[ordered]


def get_similar_songs(
    track_id: str,
    top_k: int = 10,
    preset: Optional[str] = None,
    use_genre_boost: bool = True,
    use_artist_diversity: bool = True,
) -> pd.DataFrame:
    """
    Find tracks that are similar to the given track_id in the full
    feature space (danceability/energy/valence/tempo/… depending on
    build_feature_matrix implementation).
    
    Args:
        track_id: The Spotify track ID to find similar songs for
        top_k: Number of recommendations to return
        preset: Optional feature weight preset ("mood", "workout", "chill", "psychedelic")
        use_genre_boost: If True, boost same-genre tracks
        use_artist_diversity: If True, limit max songs per artist
    """
    songs = _get_songs_df()
    X, feature_cols = _get_feature_matrix(preset=preset)

    if "track_id" not in songs.columns:
        raise KeyError("Songs dataframe must contain a 'track_id' column.")

    matches = songs.index[songs["track_id"] == track_id]
    if len(matches) == 0:
        raise KeyError(f"Unknown track_id: {track_id}")
    idx = matches[0]
    
    # Get reference track info for genre boosting
    ref_track = songs.loc[idx]
    ref_genre = ref_track.get("playlist_genre") if "playlist_genre" in songs.columns else None
    ref_subgenre = ref_track.get("playlist_subgenre") if "playlist_subgenre" in songs.columns else None

    base_vec = X[idx : idx + 1]
    sims = cosine_similarity(base_vec, X)[0]

    result = songs.copy()
    result["similarity"] = sims

    # Drop the reference track itself
    result = result[result["track_id"] != track_id]
    
    # Remove duplicate tracks (same track_id), keep the one with highest similarity
    result = result.sort_values("similarity", ascending=False)
    result = result.drop_duplicates(subset=["track_id"], keep="first")
    
    # Apply genre boosting if enabled
    if use_genre_boost:
        result = _apply_genre_boost(result, ref_genre, ref_subgenre)
        result = result.sort_values("similarity", ascending=False)
    
    # Apply artist diversity if enabled
    if use_artist_diversity:
        result = _apply_artist_diversity(result, top_k)
    else:
        result = result.head(top_k)

    def _explain(row: pd.Series) -> str:
        genre_info = ""
        if ref_genre and "playlist_genre" in row.index:
            row_genre = row.get("playlist_genre", "")
            if row_genre and str(row_genre).lower() == str(ref_genre).lower():
                genre_info = f" Both are in the {ref_genre} genre."
        return (
            f"Recommended because it has similar audio characteristics "
            f"(valence, energy, tempo, etc.) to your chosen track.{genre_info}"
        )

    result = result.copy()
    result["explanation"] = result.apply(_explain, axis=1)

    preferred_cols = [
        "track_id",
        "track_name",
        "track_artist",
        "playlist_genre",
        "playlist_subgenre",
        "similarity",
        "explanation",
    ]
    ordered = [c for c in preferred_cols if c in result.columns] + [
        c for c in result.columns if c not in preferred_cols
    ]
    return result[ordered]


def get_similar_songs_by_name(
    song_name: str,
    top_k: int = 10,
    preset: Optional[str] = None,
    use_genre_boost: bool = True,
    use_artist_diversity: bool = True,
) -> pd.DataFrame:
    """
    Find tracks similar to a song by searching for the song name (fuzzy match).
    Returns similar songs based on audio features.
    
    Args:
        song_name: Name of the song to search for
        top_k: Number of recommendations to return
        preset: Optional feature weight preset ("mood", "workout", "chill", "psychedelic")
        use_genre_boost: If True, boost same-genre tracks
        use_artist_diversity: If True, limit max songs per artist
    """
    songs = _get_songs_df()
    X, feature_cols = _get_feature_matrix(preset=preset)

    if "track_name" not in songs.columns:
        raise KeyError("Songs dataframe must contain a 'track_name' column.")

    # Case-insensitive partial match
    song_name_lower = song_name.lower().strip()
    
    # Try exact match first
    mask = songs["track_name"].str.lower().str.strip() == song_name_lower
    if mask.sum() == 0:
        # Try partial/contains match
        mask = songs["track_name"].str.lower().str.contains(song_name_lower, na=False, regex=False)
    
    if mask.sum() == 0:
        # No match found - return empty with message
        return pd.DataFrame({
            "track_name": [],
            "track_artist": [],
            "similarity": [],
            "explanation": [],
        })
    
    # Get the first matching song
    idx = songs.index[mask][0]
    matched_song = songs.loc[idx]
    matched_name = matched_song["track_name"]
    matched_artist = matched_song.get("track_artist", "Unknown")
    matched_track_id = matched_song.get("track_id", "")
    ref_genre = matched_song.get("playlist_genre") if "playlist_genre" in songs.columns else None
    ref_subgenre = matched_song.get("playlist_subgenre") if "playlist_subgenre" in songs.columns else None

    base_vec = X[idx : idx + 1]
    sims = cosine_similarity(base_vec, X)[0]

    result = songs.copy()
    result["similarity"] = sims

    # Remove the reference track (by track_id to catch duplicates)
    result = result[result["track_id"] != matched_track_id]
    
    # Remove duplicate tracks (same track_id), keep the one with highest similarity
    result = result.sort_values("similarity", ascending=False)
    result = result.drop_duplicates(subset=["track_id"], keep="first")
    
    # Apply genre boosting if enabled
    if use_genre_boost:
        result = _apply_genre_boost(result, ref_genre, ref_subgenre)
        result = result.sort_values("similarity", ascending=False)
    
    # Apply artist diversity if enabled
    if use_artist_diversity:
        result = _apply_artist_diversity(result, top_k)
    else:
        result = result.head(top_k)

    def _explain(row: pd.Series) -> str:
        genre_info = ""
        if ref_genre and "playlist_genre" in row.index:
            row_genre = row.get("playlist_genre", "")
            if row_genre and str(row_genre).lower() == str(ref_genre).lower():
                genre_info = f" Both are in the {ref_genre} genre."
        return (
            f"Similar to \"{matched_name}\" by {matched_artist} based on audio features "
            f"(danceability, energy, valence, tempo, etc.).{genre_info}"
        )

    result = result.copy()
    result["explanation"] = result.apply(_explain, axis=1)

    preferred_cols = [
        "track_id",
        "track_name",
        "track_artist",
        "playlist_genre",
        "playlist_subgenre",
        "similarity",
        "explanation",
    ]
    ordered = [c for c in preferred_cols if c in result.columns] + [
        c for c in result.columns if c not in preferred_cols
    ]
    return result[ordered]