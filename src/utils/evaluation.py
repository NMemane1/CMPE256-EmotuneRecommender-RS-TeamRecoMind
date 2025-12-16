"""
Utility functions for evaluating clustering quality and recommendation lists.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# -------------------------------------------------------------------
# Clustering evaluation
# -------------------------------------------------------------------


def compute_silhouette_for_ks(
    X: np.ndarray,
    ks: Iterable[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute silhouette scores for multiple values of k using k-means.

    Args:
        X: 2D numpy array of shape (n_samples, n_features),
           usually the standardized feature matrix (e.g. X_scaled).
        ks: Iterable of k values to try, e.g. [3, 4, 5, 6].
        random_state: Random seed for KMeans.

    Returns:
        DataFrame with columns:
        - 'k'           : number of clusters
        - 'silhouette'  : silhouette score for that k
    """
    results: List[Dict] = []
    X = np.asarray(X)

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)
        results.append({"k": int(k), "silhouette": float(sil)})

    return pd.DataFrame(results)


def summarize_clusters(
    df_features: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: Optional[List[str]] = None,
    round_digits: int = 3,
) -> pd.DataFrame:
    """
    Compute mean feature values per cluster to help interpret clusters.

    Args:
        df_features: DataFrame with one row per track and numeric feature columns.
        labels: Cluster labels (same length as df_features).
        feature_cols: List of feature column names to include. If None,
                      all numeric columns are used.
        round_digits: Number of decimal places to round to.

    Returns:
        DataFrame indexed by cluster_id with mean feature values.
    """
    if feature_cols is None:
        feature_cols = df_features.select_dtypes(include=["float64", "int64"]).columns.tolist()

    tmp = df_features.copy()
    tmp["cluster_id"] = labels

    cluster_summary = (
        tmp.groupby("cluster_id")[feature_cols]
        .mean()
        .round(round_digits)
    )
    return cluster_summary


def cluster_size_distribution(labels: np.ndarray) -> pd.Series:
    """
    Return the size of each cluster as a pandas Series.

    Args:
        labels: Cluster labels array.

    Returns:
        Series where index is cluster_id and values are counts.
    """
    return pd.Series(labels, name="cluster_id").value_counts().sort_index()


# -------------------------------------------------------------------
# Recommendation evaluation
# -------------------------------------------------------------------


def recommendation_basic_stats(rec_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute basic statistics for a recommendation list.

    Expects columns like:
    - 'track_id'
    - 'track_artist'
    - 'playlist_genre' (optional)
    - 'track_popularity' (optional)

    Args:
        rec_df: DataFrame of recommended tracks (one row per recommended song).

    Returns:
        Dictionary with metrics such as:
        - n_recs
        - n_unique_artists
        - n_unique_genres (if available)
        - mean_popularity (if available)
        - min_popularity, max_popularity (if available)
    """
    stats: Dict[str, float] = {}

    stats["n_recs"] = len(rec_df)

    if "track_artist" in rec_df.columns:
        stats["n_unique_artists"] = rec_df["track_artist"].nunique()

    if "playlist_genre" in rec_df.columns:
        stats["n_unique_genres"] = rec_df["playlist_genre"].nunique()

    if "track_popularity" in rec_df.columns:
        pop = rec_df["track_popularity"].dropna()
        if len(pop) > 0:
            stats["mean_popularity"] = float(pop.mean())
            stats["min_popularity"] = float(pop.min())
            stats["max_popularity"] = float(pop.max())

    return stats


def recommendation_diversity_metrics(rec_df: pd.DataFrame) -> Dict[str, float]:
    """
    Simple diversity metrics for a recommendation list.

    Args:
        rec_df: DataFrame of recommended tracks.

    Returns:
        Dictionary with metrics such as:
        - artist_entropy
        - genre_entropy (if genres available)
        - artist_coverage (unique artists / total recs)
        - genre_coverage (unique genres / total recs)
    """
    metrics: Dict[str, float] = {}
    n = len(rec_df)
    if n == 0:
        return metrics

    # Artist diversity
    if "track_artist" in rec_df.columns:
        counts = rec_df["track_artist"].value_counts(normalize=True)
        # Shannon entropy in bits
        artist_entropy = float(-np.sum(counts * np.log2(counts)))
        metrics["artist_entropy"] = artist_entropy
        metrics["artist_coverage"] = float(rec_df["track_artist"].nunique() / n)

    # Genre diversity
    if "playlist_genre" in rec_df.columns:
        g_counts = rec_df["playlist_genre"].value_counts(normalize=True)
        genre_entropy = float(-np.sum(g_counts * np.log2(g_counts)))
        metrics["genre_entropy"] = genre_entropy
        metrics["genre_coverage"] = float(rec_df["playlist_genre"].nunique() / n)

    return metrics


def explain_recommendation_list(
    rec_df: pd.DataFrame,
    max_rows: int = 5,
) -> pd.DataFrame:
    """
    Return a small summary table of recommended songs with key columns.
    Useful for printing in notebooks when analyzing recommendation quality.

    Args:
        rec_df: DataFrame with recommendations.
        max_rows: Number of rows to show.

    Returns:
        A smaller DataFrame with human-friendly columns.
    """
    columns_in_order = [
        "track_name",
        "track_artist",
        "playlist_genre",
        "track_popularity",
        "valence",
        "energy",
        "instrumentalness",
        "speechiness",
        "similarity",
        "explanation",
    ]
    cols = [c for c in columns_in_order if c in rec_df.columns]
    return rec_df[cols].head(max_rows)

