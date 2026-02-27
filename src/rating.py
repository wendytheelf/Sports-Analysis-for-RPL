"""
Baseline team rating from regular-season results.
Uses per-team aggregates and shrinkage toward 0 for low game count.
"""
from typing import Optional

import pandas as pd

from src.io_utils import TRAIN_HOME_SCORE, TRAIN_AWAY_SCORE, TRAIN_HOME_TEAM, TRAIN_AWAY_TEAM


def build_team_aggregates(train: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team aggregates from train DataFrame (canonical columns).
    Returns DataFrame with: team, games_played, wins, losses, win_pct,
    avg_points_for, avg_points_against, avg_margin,
    home_games, away_games, home_avg_margin, away_avg_margin.
    """
    rows = []
    all_teams = pd.concat(
        [train[TRAIN_HOME_TEAM].astype(str), train[TRAIN_AWAY_TEAM].astype(str)],
        ignore_index=True,
    )
    teams = sorted(all_teams.unique())
    teams = [t for t in teams if t and str(t).strip() != ""]

    for team in teams:
        home = train[train[TRAIN_HOME_TEAM].astype(str) == team]
        away = train[train[TRAIN_AWAY_TEAM].astype(str) == team]
        home_games = len(home)
        away_games = len(away)
        games_played = home_games + away_games

        home_pts_for = home[TRAIN_HOME_SCORE].sum()
        home_pts_against = home[TRAIN_AWAY_SCORE].sum()
        away_pts_for = away[TRAIN_AWAY_SCORE].sum()
        away_pts_against = away[TRAIN_HOME_SCORE].sum()
        points_for = home_pts_for + away_pts_for
        points_against = home_pts_against + away_pts_against

        home_margins = (home[TRAIN_HOME_SCORE] - home[TRAIN_AWAY_SCORE]).values
        away_margins = (away[TRAIN_AWAY_SCORE] - away[TRAIN_HOME_SCORE]).values
        home_avg_margin = home_margins.mean() if len(home_margins) > 0 else 0.0
        away_avg_margin = away_margins.mean() if len(away_margins) > 0 else 0.0

        wins = (home[TRAIN_HOME_SCORE] > home[TRAIN_AWAY_SCORE]).sum() + (
            away[TRAIN_AWAY_SCORE] > away[TRAIN_HOME_SCORE]
        ).sum()
        losses = games_played - wins
        win_pct = wins / games_played if games_played > 0 else 0.0
        avg_margin = (points_for - points_against) / games_played if games_played > 0 else 0.0

        rows.append(
            {
                "team": team,
                "games_played": games_played,
                "wins": int(wins),
                "losses": int(losses),
                "win_pct": win_pct,
                "avg_points_for": points_for / games_played if games_played > 0 else 0.0,
                "avg_points_against": points_against / games_played if games_played > 0 else 0.0,
                "avg_margin": avg_margin,
                "home_games": home_games,
                "away_games": away_games,
                "home_avg_margin": home_avg_margin,
                "away_avg_margin": away_avg_margin,
            }
        )

    return pd.DataFrame(rows)


def add_ratings(
    team_stats: pd.DataFrame,
    alpha: float = 5.0,
) -> pd.DataFrame:
    """
    Add baseline rating and rating_shrunk to team_stats.
    rating = avg_margin.
    rating_shrunk = (games_played / (games_played + alpha)) * avg_margin.
    """
    out = team_stats.copy()
    out["rating"] = out["avg_margin"]
    gp = out["games_played"]
    out["rating_shrunk"] = (gp / (gp + alpha)) * out["avg_margin"]
    return out


def compute_ratings(
    train: pd.DataFrame,
    alpha: float = 5.0,
) -> pd.DataFrame:
    """
    Full pipeline: build team aggregates and add baseline (shrunk) ratings.
    """
    agg = build_team_aggregates(train)
    return add_ratings(agg, alpha=alpha)
