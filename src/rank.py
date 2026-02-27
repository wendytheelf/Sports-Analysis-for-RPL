"""
Rank all teams by baseline rating and produce Rankings.xlsx.
"""
from pathlib import Path

import pandas as pd


def build_rankings(team_stats: pd.DataFrame, rating_col: str = "rating_shrunk") -> pd.DataFrame:
    """
    Build rankings DataFrame: Rank (1..N), Team, Rating.
    Rank 1 = best (highest rating).
    """
    df = (
        team_stats[["team", rating_col]]
        .rename(columns={rating_col: "Rating", "team": "Team"})
        .sort_values("Rating", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def save_rankings(rankings: pd.DataFrame, path: Path) -> None:
    """Save rankings to Excel (Rankings.xlsx)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_excel(path, index=False, sheet_name="Rankings")


def load_rankings(path: Path) -> pd.DataFrame:
    """Load rankings from Excel."""
    return pd.read_excel(path)
