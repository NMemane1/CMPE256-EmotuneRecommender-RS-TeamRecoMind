from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
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


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Return a numpy array of the numerical features and the list of feature column names.
    """
    X = df[AUDIO_FEATURES].astype(float).values
    return X, AUDIO_FEATURES