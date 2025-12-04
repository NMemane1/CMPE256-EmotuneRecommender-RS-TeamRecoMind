from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

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


def load_song_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the Spotify songs dataset and validate required columns.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_META + AUDIO_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in songs dataset: {missing}")

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame containing only the numerical features we use for similarity.
    """
    return df[AUDIO_FEATURES].astype(float)