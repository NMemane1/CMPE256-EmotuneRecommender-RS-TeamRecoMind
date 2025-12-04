from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Path to your dataset
DATA_PATH = Path("data/songs.csv")

# Numerical audio features we will use for similarity
AUDIO_FEATURES: List[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Required metadata columns
REQUIRED_META = ["track_id", "track_name", "track_artist"]

# Optional metadata for genre filtering
GENRE_COLS = ["playlist_genre", "playlist_subgenre"]

# --------------------------------------------------------------------
# Feature Weight Presets
# --------------------------------------------------------------------
# Keys are feature names, values are multiplicative weights.
# Higher weight = more influence on similarity.
FEATURE_WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "default": {f: 1.0 for f in AUDIO_FEATURES},
    "mood": {
        "danceability": 0.8,
        "energy": 1.5,
        "loudness": 0.5,
        "speechiness": 0.3,
        "acousticness": 1.2,
        "instrumentalness": 1.0,
        "liveness": 0.3,
        "valence": 1.5,
        "tempo": 0.6,
    },
    "workout": {
        "danceability": 1.5,
        "energy": 2.0,
        "loudness": 1.0,
        "speechiness": 0.3,
        "acousticness": 0.3,
        "instrumentalness": 0.5,
        "liveness": 0.5,
        "valence": 0.8,
        "tempo": 1.5,
    },
    "chill": {
        "danceability": 0.5,
        "energy": 0.5,
        "loudness": 0.5,
        "speechiness": 0.3,
        "acousticness": 1.8,
        "instrumentalness": 1.5,
        "liveness": 0.3,
        "valence": 1.2,
        "tempo": 0.5,
    },
    # Psychedelic/dreamy: prioritize instrumentalness, valence (pleasant), moderate tempo
    # De-emphasize raw energy (which can be misleading for dense productions)
    "psychedelic": {
        "danceability": 1.0,
        "energy": 0.4,  # Lower weight - energy metrics can be misleading for psychedelic
        "loudness": 0.3,
        "speechiness": 0.2,
        "acousticness": 1.5,
        "instrumentalness": 2.0,  # High weight - psychedelic often instrumental/textural
        "liveness": 0.3,
        "valence": 1.8,  # High weight - match the pleasant/dreamy vibe
        "tempo": 1.2,
    },
    # Indie/alternative: balanced but less EDM-centric
    "indie": {
        "danceability": 0.8,
        "energy": 0.7,
        "loudness": 0.5,
        "speechiness": 0.5,
        "acousticness": 1.5,
        "instrumentalness": 1.2,
        "liveness": 0.8,
        "valence": 1.3,
        "tempo": 1.0,
    },
}

# Global scaler (fitted once on load)
_SCALER: Optional[StandardScaler] = None


def load_song_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the Spotify songs dataset and validate required columns.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_META + AUDIO_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in songs dataset: {missing}")

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    preset: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return a numpy array of the numerical features (scaled) and the list of feature column names.
    
    Args:
        df: Songs dataframe
        weights: Optional dict of feature_name -> weight (multiplier)
        preset: Optional preset name from FEATURE_WEIGHT_PRESETS (e.g., "mood", "workout", "chill")
    
    Returns:
        Tuple of (feature_matrix, feature_column_names)
    """
    global _SCALER
    
    X = df[AUDIO_FEATURES].astype(float).values
    
    # Handle NaN/inf values before scaling
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Apply StandardScaler to normalize features (so tempo/loudness don't dominate)
    if _SCALER is None:
        _SCALER = StandardScaler()
        X_scaled = _SCALER.fit_transform(X)
    else:
        X_scaled = _SCALER.transform(X)
    
    # Handle any NaN/inf that might arise from scaling
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply feature weights if provided
    if preset and preset in FEATURE_WEIGHT_PRESETS:
        weights = FEATURE_WEIGHT_PRESETS[preset]
    
    if weights:
        weight_array = np.array([weights.get(f, 1.0) for f in AUDIO_FEATURES])
        X_scaled = X_scaled * weight_array
    
    return X_scaled, AUDIO_FEATURES


def get_genre_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract genre columns if available.
    """
    genre_cols_present = [c for c in GENRE_COLS if c in df.columns]
    if genre_cols_present:
        return df[genre_cols_present]
    return pd.DataFrame(index=df.index)