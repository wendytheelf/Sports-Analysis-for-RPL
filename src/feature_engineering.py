"""
Feature engineering for RPL baseline pipeline.
Extend this module to add new features (e.g., strength of schedule, recent form).
"""
import pandas as pd


def add_train_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the training DataFrame.
    Baseline: no extra columns. Override or extend for custom features.
    """
    return df.copy()


def add_derby_features(
    derby_df: pd.DataFrame,
    team_stats: pd.DataFrame,
    rating_col: str = "rating_shrunk",
) -> pd.DataFrame:
    """
    Add features to derby/matchup DataFrame by joining team stats.
    Baseline: only rating. Extend to add head-to-head, venue, etc.
    """
    out = derby_df.copy()
    # Join will be done in predict.py using team_stats; this hook is for extra cols.
    return out
