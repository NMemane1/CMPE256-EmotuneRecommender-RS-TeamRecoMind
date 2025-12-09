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
# Full 9-Feature Mood Prototypes
# --------------------------------------------------------------------
# Each mood is defined as a target vector across ALL audio features.
# This is much more accurate than just valence+energy.
#
# Features (in order): danceability, energy, loudness, speechiness,
#                      acousticness, instrumentalness, liveness, valence, tempo
#
# Values are normalized targets (0-1 scale, tempo scaled to 0-1 from 0-200 BPM)
from src.recommender.feature_builder import AUDIO_FEATURES

MOOD_PROTOTYPES: Dict[str, Dict[str, float]] = {
    "happy": {
        "danceability": 0.70,
        "energy": 0.75,
        "loudness": 0.65,  # Normalized: -60 to 0 dB → 0 to 1
        "speechiness": 0.10,
        "acousticness": 0.25,
        "instrumentalness": 0.05,
        "liveness": 0.20,
        "valence": 0.85,
        "tempo": 0.60,  # ~120 BPM
    },
    "sad": {
        "danceability": 0.35,
        "energy": 0.30,
        "loudness": 0.40,
        "speechiness": 0.05,
        "acousticness": 0.70,
        "instrumentalness": 0.20,
        "liveness": 0.10,
        "valence": 0.20,
        "tempo": 0.40,  # ~80 BPM, slower
    },
    "energetic": {
        "danceability": 0.80,
        "energy": 0.90,
        "loudness": 0.80,
        "speechiness": 0.15,
        "acousticness": 0.10,
        "instrumentalness": 0.10,
        "liveness": 0.30,
        "valence": 0.75,
        "tempo": 0.70,  # ~140 BPM
    },
    "calm": {
        "danceability": 0.40,
        "energy": 0.25,
        "loudness": 0.35,
        "speechiness": 0.05,
        "acousticness": 0.75,
        "instrumentalness": 0.40,
        "liveness": 0.10,
        "valence": 0.55,
        "tempo": 0.35,  # ~70 BPM
    },
    "angry": {
        "danceability": 0.50,
        "energy": 0.90,
        "loudness": 0.85,
        "speechiness": 0.20,
        "acousticness": 0.05,
        "instrumentalness": 0.05,
        "liveness": 0.25,
        "valence": 0.25,
        "tempo": 0.65,  # ~130 BPM
    },
    "romantic": {
        "danceability": 0.55,
        "energy": 0.45,
        "loudness": 0.50,
        "speechiness": 0.05,
        "acousticness": 0.60,
        "instrumentalness": 0.15,
        "liveness": 0.15,
        "valence": 0.75,
        "tempo": 0.45,  # ~90 BPM
    },
    "mellow": {
        "danceability": 0.50,
        "energy": 0.40,
        "loudness": 0.45,
        "speechiness": 0.05,
        "acousticness": 0.55,
        "instrumentalness": 0.30,
        "liveness": 0.12,
        "valence": 0.55,
        "tempo": 0.45,  # ~90 BPM
    },
    "focus": {
        # Ideal for studying: instrumental, low speech, moderate tempo, not too energetic
        "danceability": 0.45,
        "energy": 0.35,
        "loudness": 0.40,
        "speechiness": 0.03,  # Very low - no distracting vocals
        "acousticness": 0.50,
        "instrumentalness": 0.70,  # High - instrumental music for focus
        "liveness": 0.10,
        "valence": 0.50,
        "tempo": 0.50,  # ~100 BPM - steady, not too fast
    },
    "nostalgic": {
        # Nostalgic/throwback vibes - year boost applied separately
        "danceability": 0.55,
        "energy": 0.50,
        "loudness": 0.50,
        "speechiness": 0.08,
        "acousticness": 0.45,
        "instrumentalness": 0.15,
        "liveness": 0.15,
        "valence": 0.60,  # Bittersweet positive
        "tempo": 0.50,
    },
}

# Legacy 2D targets for backward compatibility
MOOD_TARGETS: Dict[str, Dict[str, float]] = {
    mood: {"valence": proto["valence"], "energy": proto["energy"]}
    for mood, proto in MOOD_PROTOTYPES.items()
}


def _normalize_tempo(tempo: float) -> float:
    """Normalize tempo from BPM (typically 50-200) to 0-1 scale."""
    return np.clip((tempo - 50) / 150, 0, 1)


def _get_mood_prototype_vector(mood: str) -> np.ndarray:
    """Get the prototype vector for a mood in the same order as AUDIO_FEATURES."""
    mood_key = mood.lower()
    proto = MOOD_PROTOTYPES.get(mood_key, MOOD_PROTOTYPES["calm"])
    return np.array([proto.get(f, 0.5) for f in AUDIO_FEATURES])


# --------------------------------------------------------------------
# Public API – used by recommendation_pipeline.py
# --------------------------------------------------------------------
def get_mood_recommendations(mood: str, top_k: int = 10) -> pd.DataFrame:
    """
    Return top_k tracks that best match the requested mood using
    cosine similarity across ALL 9 audio features (not just valence+energy).
    
    This provides much more accurate mood matching by considering:
    - instrumentalness (important for focus/study)
    - speechiness (vocals can be distracting)
    - tempo (energy level)
    - acousticness (electronic vs acoustic vibe)
    - and more...
    """
    songs = _get_songs_df()
    X, feature_cols = _get_feature_matrix()  # Uses StandardScaler
    
    # Get the mood prototype and scale it the same way
    mood_key = mood.lower()
    if mood_key not in MOOD_PROTOTYPES:
        mood_key = "calm"  # Safe fallback
    
    proto = MOOD_PROTOTYPES[mood_key]
    
    # Build prototype vector in raw feature space, then we'll compare
    # We need to handle tempo specially (it's in BPM, not 0-1)
    proto_raw = []
    for f in AUDIO_FEATURES:
        if f == "tempo":
            # Convert our 0-1 tempo target back to BPM for comparison
            proto_raw.append(proto.get(f, 0.5) * 150 + 50)
        elif f == "loudness":
            # Convert our 0-1 loudness target back to dB (roughly -60 to 0)
            proto_raw.append(proto.get(f, 0.5) * 60 - 60)
        else:
            proto_raw.append(proto.get(f, 0.5))
    
    proto_vector = np.array(proto_raw).reshape(1, -1)
    
    # Scale the prototype using the same scaler as the songs
    from src.recommender.feature_builder import _SCALER
    if _SCALER is not None:
        proto_scaled = _SCALER.transform(proto_vector)
    else:
        proto_scaled = proto_vector
    
    # Compute cosine similarity between prototype and all songs
    sims = cosine_similarity(proto_scaled, X)[0]
    
    # Add similarity scores to dataframe
    result = songs.copy()
    result["similarity"] = sims
    
    # Apply year-based boost ONLY for nostalgic mood (prefer older songs)
    # For other moods, year doesn't improve recommendations
    if mood_key == "nostalgic" and "track_album_release_date" in result.columns:
        try:
            # Extract year from release date
            result["_year"] = pd.to_datetime(
                result["track_album_release_date"], errors="coerce"
            ).dt.year
            
            # Boost songs from before 2010 by up to 10%
            # Older = more nostalgic feel
            year_boost = result["_year"].apply(
                lambda y: 1.08 if pd.notna(y) and y < 2000 else 
                         (1.05 if pd.notna(y) and y < 2010 else 1.0)
            )
            result["similarity"] = result["similarity"] * year_boost
            result = result.drop(columns=["_year"])
        except Exception:
            pass  # If year parsing fails, just skip the boost
    
    # Sort by similarity and take top_k
    result = result.sort_values("similarity", ascending=False).head(top_k)
    
    # Generate explanations highlighting key matching features
    def _explain(row: pd.Series) -> str:
        highlights = []
        if mood_key == "focus":
            inst = row.get("instrumentalness", 0)
            speech = row.get("speechiness", 0)
            if inst > 0.3:
                highlights.append(f"instrumental ({inst:.0%})")
            if speech < 0.1:
                highlights.append("low vocals")
        
        v = row.get("valence", np.nan)
        e = row.get("energy", np.nan)
        base = f"Matches '{mood}' mood (valence={v:.2f}, energy={e:.2f})"
        
        if highlights:
            return base + " — " + ", ".join(highlights)
        return base
    
    result = result.copy()
    result["explanation"] = result.apply(_explain, axis=1)
    
    preferred_cols = [
        "track_id",
        "track_name",
        "track_artist",
        "valence",
        "energy",
        "instrumentalness",
        "speechiness",
        "similarity",
        "explanation",
    ]
    ordered = [c for c in preferred_cols if c in result.columns] + [
        c for c in result.columns if c not in preferred_cols
    ]
    return result[ordered]


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