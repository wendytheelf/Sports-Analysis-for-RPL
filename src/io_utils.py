"""
I/O utilities with robust schema inference for Train and Derby/Test CSVs.
Infers column names for home/away teams, scores, and derby matchups.
"""
from pathlib import Path
from typing import Optional

import pandas as pd


# Canonical names we use internally
TRAIN_HOME_TEAM = "home_team"
TRAIN_AWAY_TEAM = "away_team"
TRAIN_HOME_SCORE = "home_score"
TRAIN_AWAY_SCORE = "away_score"

DERBY_TEAM1 = "team1"
DERBY_TEAM2 = "team2"
DERBY_MATCH_ID = "match_id"

# Possible column name variants (case-insensitive match)
HOME_TEAM_VARIANTS = ("hometeam", "home_team", "home team")
AWAY_TEAM_VARIANTS = ("awayteam", "away_team", "away team")
HOME_SCORE_VARIANTS = ("homepts", "home_pts", "home score", "homescore", "home_score")
AWAY_SCORE_VARIANTS = ("awaypts", "away_pts", "away score", "awayscore", "away_score")

TEAM1_VARIANTS = ("team1", "teama", "home", "team_a")
TEAM2_VARIANTS = ("team2", "teamb", "away", "team_b")
MATCH_ID_VARIANTS = ("gameid", "matchid", "id", "game_id", "match_id")


def _normalize(col: str) -> str:
    """Normalize column name for matching: lowercase, strip, replace spaces with underscore."""
    return col.strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(df: pd.DataFrame, variants: tuple[str, ...]) -> Optional[str]:
    """Return first DataFrame column that matches any variant, or None."""
    cols = {_normalize(c): c for c in df.columns}
    for v in variants:
        if v in cols:
            return cols[v]
    return None


def infer_train_schema(df: pd.DataFrame) -> dict[str, str]:
    """
    Infer train CSV schema. Returns mapping from canonical name to actual column name.
    Raises ValueError if required columns cannot be inferred.
    """
    mapping = {}
    home_team = _find_column(df, HOME_TEAM_VARIANTS)
    away_team = _find_column(df, AWAY_TEAM_VARIANTS)
    home_score = _find_column(df, HOME_SCORE_VARIANTS)
    away_score = _find_column(df, AWAY_SCORE_VARIANTS)

    if home_team is None:
        raise ValueError(
            "Train CSV: could not find home team column. "
            f"Expected one of (case-insensitive): {HOME_TEAM_VARIANTS}. "
            f"Found columns: {list(df.columns)}"
        )
    if away_team is None:
        raise ValueError(
            "Train CSV: could not find away team column. "
            f"Expected one of: {AWAY_TEAM_VARIANTS}. Found: {list(df.columns)}"
        )
    if home_score is None:
        raise ValueError(
            "Train CSV: could not find home score column. "
            f"Expected one of: {HOME_SCORE_VARIANTS}. Found: {list(df.columns)}"
        )
    if away_score is None:
        raise ValueError(
            "Train CSV: could not find away score column. "
            f"Expected one of: {AWAY_SCORE_VARIANTS}. Found: {list(df.columns)}"
        )

    mapping[TRAIN_HOME_TEAM] = home_team
    mapping[TRAIN_AWAY_TEAM] = away_team
    mapping[TRAIN_HOME_SCORE] = home_score
    mapping[TRAIN_AWAY_SCORE] = away_score
    return mapping


def infer_derby_schema(df: pd.DataFrame) -> dict[str, str]:
    """
    Infer derby/test CSV schema (matchups without scores).
    Returns mapping from canonical name to actual column name.
    """
    mapping = {}
    team1 = _find_column(df, TEAM1_VARIANTS)
    team2 = _find_column(df, TEAM2_VARIANTS)
    match_id = _find_column(df, MATCH_ID_VARIANTS)

    if team1 is None:
        raise ValueError(
            "Derby CSV: could not find team1 column. "
            f"Expected one of (case-insensitive): {TEAM1_VARIANTS}. "
            f"Found columns: {list(df.columns)}"
        )
    if team2 is None:
        raise ValueError(
            "Derby CSV: could not find team2 column. "
            f"Expected one of: {TEAM2_VARIANTS}. Found: {list(df.columns)}"
        )

    mapping[DERBY_TEAM1] = team1
    mapping[DERBY_TEAM2] = team2
    if match_id:
        mapping[DERBY_MATCH_ID] = match_id
    return mapping


def load_train_csv(path: Path) -> pd.DataFrame:
    """
    Load train CSV and return DataFrame with canonical column names
    (home_team, away_team, home_score, away_score). Preserves all original columns.
    """
    df = pd.read_csv(path)
    schema = infer_train_schema(df)
    out = df.copy()
    out[TRAIN_HOME_TEAM] = df[schema[TRAIN_HOME_TEAM]]
    out[TRAIN_AWAY_TEAM] = df[schema[TRAIN_AWAY_TEAM]]
    out[TRAIN_HOME_SCORE] = pd.to_numeric(df[schema[TRAIN_HOME_SCORE]], errors="coerce")
    out[TRAIN_AWAY_SCORE] = pd.to_numeric(df[schema[TRAIN_AWAY_SCORE]], errors="coerce")
    return out


def load_derby_csv(path: Path) -> pd.DataFrame:
    """
    Load derby/test CSV and return DataFrame with canonical columns
    (team1, team2, and optionally match_id). Preserves all original columns.
    """
    df = pd.read_csv(path)
    schema = infer_derby_schema(df)
    out = df.copy()
    out[DERBY_TEAM1] = df[schema[DERBY_TEAM1]]
    out[DERBY_TEAM2] = df[schema[DERBY_TEAM2]]
    if DERBY_MATCH_ID in schema:
        out[DERBY_MATCH_ID] = df[schema[DERBY_MATCH_ID]]
    return out


def detect_derby_file(data_dir: Path) -> Optional[Path]:
    """Return path to first existing derby/test file in data_dir, or None."""
    from config import DEFAULT_DERBY_CANDIDATES

    for name in DEFAULT_DERBY_CANDIDATES:
        p = data_dir / name
        if p.exists():
            return p
    return None
