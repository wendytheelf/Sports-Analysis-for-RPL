"""
Derby match prediction: winner and margin from baseline ratings.
Neutral site: no home advantage; margin = rating(team1) - rating(team2).
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from src.io_utils import DERBY_TEAM1, DERBY_TEAM2, DERBY_MATCH_ID


def predict_derbies(
    derby_df: pd.DataFrame,
    team_stats: pd.DataFrame,
    rating_col: str = "rating_shrunk",
) -> pd.DataFrame:
    """
    Predict winner and margin for each derby row.
    Predicted margin = rating(team1) - rating(team2) (neutral site).
    Predicted winner = team1 if margin >= 0 else team2; tie -> team1, margin 0.
    PredictedMargin = absolute margin (victory margin, non-negative).
    """
    rating_map = team_stats.set_index("team")[rating_col].to_dict()

    def get_rating(team: str) -> float:
        return rating_map.get(str(team).strip(), 0.0)

    out = derby_df.copy()
    r1 = out[DERBY_TEAM1].map(get_rating)
    r2 = out[DERBY_TEAM2].map(get_rating)
    margin = r1 - r2  # from team1's perspective
    out["PredictedWinner"] = out[DERBY_TEAM1].where(margin >= 0, out[DERBY_TEAM2])
    out["PredictedMargin"] = margin.abs().round(2)
    return out


def build_predictions_csv(
    predictions: pd.DataFrame,
    original_derby_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build submission-ready predictions DataFrame.
    Preserves match id (GameID/MatchID), Team1/Team2, adds PredictedWinner, PredictedMargin.
    If derby file has Team1_WinMargin, we can output margin from Team1's perspective.
    """
    out = predictions.copy()
    # Ensure common column names for submission
    if "PredictedMargin" not in out.columns:
        raise ValueError("predictions must contain PredictedMargin")
    # Team1_WinMargin: margin from Team1's perspective (positive = Team1 wins)
    r1 = out[DERBY_TEAM1]
    r2 = out[DERBY_TEAM2]
    winner = out["PredictedWinner"]
    margin_abs = out["PredictedMargin"]
    team1_wins = (winner == r1).astype(int)
    out["Team1_WinMargin"] = (2 * team1_wins - 1) * margin_abs
    return out


def save_predictions_csv(predictions: pd.DataFrame, path: Path) -> None:
    """Save predictions to CSV with submission schema: GameID, Team1, Team2, PredictedWinner, PredictedMargin, Team1_WinMargin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    export = predictions.copy()
    # Map canonical to submission names
    if DERBY_MATCH_ID in export.columns:
        export["GameID"] = export[DERBY_MATCH_ID]
    if DERBY_TEAM1 in export.columns:
        export["Team1"] = export[DERBY_TEAM1]
    if DERBY_TEAM2 in export.columns:
        export["Team2"] = export[DERBY_TEAM2]
    if "Team1_WinMargin" not in export.columns:
        export = build_predictions_csv(export)
    out_cols = [c for c in ["GameID", "Team1", "Team2", "PredictedWinner", "PredictedMargin", "Team1_WinMargin"] if c in export.columns]
    export[out_cols].to_csv(path, index=False)
