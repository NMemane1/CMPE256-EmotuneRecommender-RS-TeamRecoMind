from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.recommender.feature_builder import load_song_data, build_feature_matrix


# --------------------------------------------------------------------
# Global in-memory cache for songs and feature matrix
# --------------------------------------------------------------------
_SONGS_DF: pd.DataFrame | None = None
_FEATURE_MATRIX: np.ndarray | None = None
_FEATURE_COLS: list[str] | None = None


def _get_songs_df() -> pd.DataFrame:
    """Load songs.csv once and cache it."""
    global _SONGS_DF
    if _SONGS_DF is None:
        _SONGS_DF = load_song_data()
    return _SONGS_DF


def _get_feature_matrix() -> tuple[np.ndarray, list[str]]:
    """
    Use build_feature_matrix from feature_builder so whatever you defined
    there (danceability/energy/valence/tempo/etc.) is reused consistently.
    """
    global _FEATURE_MATRIX, _FEATURE_COLS

    if _FEATURE_MATRIX is None or _FEATURE_COLS is None:
        songs = _get_songs_df()
        X, feature_cols = build_feature_matrix(songs)
        _FEATURE_MATRIX = X
        _FEATURE_COLS = feature_cols

    return _FEATURE_MATRIX, _FEATURE_COLS


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


def get_similar_songs(track_id: str, top_k: int = 10) -> pd.DataFrame:
    """
    Find tracks that are similar to the given track_id in the full
    feature space (danceability/energy/valence/tempo/… depending on
    build_feature_matrix implementation).
    """
    songs = _get_songs_df()
    X, feature_cols = _get_feature_matrix()

    if "track_id" not in songs.columns:
        raise KeyError("Songs dataframe must contain a 'track_id' column.")

    matches = songs.index[songs["track_id"] == track_id]
    if len(matches) == 0:
        raise KeyError(f"Unknown track_id: {track_id}")
    idx = matches[0]

    base_vec = X[idx : idx + 1]
    sims = cosine_similarity(base_vec, X)[0]

    result = songs.copy()
    result["similarity"] = sims

    # Drop the reference track itself, then sort
    result = result[result["track_id"] != track_id]
    result = result.sort_values("similarity", ascending=False).head(top_k)

    def _explain(row: pd.Series) -> str:
        return (
            "Recommended because it is close to your chosen track in the "
            "audio feature space (higher similarity means more similar sound)."
        )

    result = result.copy()
    result["explanation"] = result.apply(_explain, axis=1)

    preferred_cols = [
        "track_id",
        "track_name",
        "track_artist",
        "similarity",
        "explanation",
    ]
    ordered = [c for c in preferred_cols if c in result.columns] + [
        c for c in result.columns if c not in result.columns
    ]
    return result[ordered]


def get_similar_songs_by_name(song_name: str, top_k: int = 10) -> pd.DataFrame:
    """
    Find tracks similar to a song by searching for the song name (fuzzy match).
    Returns similar songs based on audio features.
    """
    songs = _get_songs_df()
    X, feature_cols = _get_feature_matrix()

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

    base_vec = X[idx : idx + 1]
    sims = cosine_similarity(base_vec, X)[0]

    result = songs.copy()
    result["similarity"] = sims

    # Remove the reference track (by track_id to catch duplicates)
    result = result[result["track_id"] != matched_track_id]
    
    # Remove duplicate tracks (same track_id), keep the one with highest similarity
    result = result.sort_values("similarity", ascending=False)
    result = result.drop_duplicates(subset=["track_id"], keep="first")
    result = result.head(top_k)

    def _explain(row: pd.Series) -> str:
        return (
            f"Similar to \"{matched_name}\" by {matched_artist} based on audio features "
            f"(danceability, energy, valence, tempo, etc.)."
        )

    result = result.copy()
    result["explanation"] = result.apply(_explain, axis=1)

    preferred_cols = [
        "track_id",
        "track_name",
        "track_artist",
        "similarity",
        "explanation",
    ]
    ordered = [c for c in preferred_cols if c in result.columns] + [
        c for c in result.columns if c not in preferred_cols
    ]
    return result[ordered]