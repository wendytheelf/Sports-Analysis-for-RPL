"""
Baseline team rating from regular-season results.
Uses per-team aggregates and shrinkage toward 0 for low game count.
Adds optional Strength of Schedule (SoS) adjusted rating_sos.
"""
from typing import Optional

import pandas as pd

from src.io_utils import (
    TRAIN_HOME_SCORE,
    TRAIN_AWAY_SCORE,
    TRAIN_HOME_TEAM,
    TRAIN_AWAY_TEAM,
    TRAIN_HOME_ID,
    TRAIN_AWAY_ID,
)


def build_team_aggregates(train: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-team aggregates from train DataFrame (canonical columns).
    Returns DataFrame with: team_id, team, games_played, wins, losses, win_pct,
    avg_points_for, avg_points_against, avg_margin,
    home_games, away_games, home_avg_margin, away_avg_margin.
    Uses team_id when HomeID/AwayID present; else team name as id.
    """
    rows = []
    has_id = TRAIN_HOME_ID in train.columns and TRAIN_AWAY_ID in train.columns
    if has_id:
        home_pairs = train[[TRAIN_HOME_ID, TRAIN_HOME_TEAM]].drop_duplicates()
        away_pairs = train[[TRAIN_AWAY_ID, TRAIN_AWAY_TEAM]].drop_duplicates()
        home_pairs = home_pairs.rename(columns={TRAIN_HOME_ID: "tid", TRAIN_HOME_TEAM: "tname"})
        away_pairs = away_pairs.rename(columns={TRAIN_AWAY_ID: "tid", TRAIN_AWAY_TEAM: "tname"})
        pairs = pd.concat([home_pairs, away_pairs]).drop_duplicates(subset=["tid"])
        team_list = list(pairs.itertuples(index=False, name=None))  # [(tid, tname), ...]
    else:
        all_teams = pd.concat(
            [train[TRAIN_HOME_TEAM].astype(str), train[TRAIN_AWAY_TEAM].astype(str)],
            ignore_index=True,
        )
        teams = sorted(all_teams.unique())
        teams = [t for t in teams if t and str(t).strip() != ""]
        team_list = [(t, t) for t in teams]

    for team_id, team in team_list:
        team = str(team)
        if has_id:
            home = train[train[TRAIN_HOME_ID] == team_id]
            away = train[train[TRAIN_AWAY_ID] == team_id]
        else:
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
                "team_id": team_id,
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


def add_sos_rating(
    team_stats: pd.DataFrame,
    train: pd.DataFrame,
    sos_weight: float = 0.5,
    sos_iters: int = 5,
) -> pd.DataFrame:
    """
    Add Strength of Schedule (SoS) adjusted rating_sos to team_stats.

    Iterative update: strength = rating_shrunk + sos_weight * mean(opponent strength).
    Repeat sos_iters times. Uses team_id for joins when available.
    """
    out = team_stats.copy()
    has_id = TRAIN_HOME_ID in train.columns and TRAIN_AWAY_ID in train.columns
    id_col = "team_id" if (has_id and "team_id" in out.columns) else "team"

    # Build long-form (team, opponent) for each game
    if has_id and "team_id" in out.columns:
        home_games = train[[TRAIN_HOME_ID, TRAIN_AWAY_ID]].rename(
            columns={TRAIN_HOME_ID: "team", TRAIN_AWAY_ID: "opponent"}
        )
        away_games = train[[TRAIN_AWAY_ID, TRAIN_HOME_ID]].rename(
            columns={TRAIN_AWAY_ID: "team", TRAIN_HOME_ID: "opponent"}
        )
    else:
        home_games = train[[TRAIN_HOME_TEAM, TRAIN_AWAY_TEAM]].rename(
            columns={TRAIN_HOME_TEAM: "team", TRAIN_AWAY_TEAM: "opponent"}
        )
        away_games = train[[TRAIN_AWAY_TEAM, TRAIN_HOME_TEAM]].rename(
            columns={TRAIN_AWAY_TEAM: "team", TRAIN_HOME_TEAM: "opponent"}
        )
    games = pd.concat([home_games, away_games], ignore_index=True)

    # Index by same key as team_stats
    strength = out.set_index(id_col)["rating_shrunk"].copy()

    for _ in range(sos_iters):
        opp_strength = games.merge(
            strength.rename("opp_strength"),
            left_on="opponent",
            right_index=True,
            how="left",
        )
        opponent_avg = opp_strength.groupby("team")["opp_strength"].mean()
        strength = out.set_index(id_col)["rating_shrunk"] + sos_weight * opponent_avg.reindex(strength.index).fillna(0)

    out["rating_sos"] = out[id_col].map(strength).values
    return out


def compute_ratings(
    train: pd.DataFrame,
    alpha: float = 5.0,
    sos_weight: float = 0.5,
    sos_iters: int = 5,
) -> pd.DataFrame:
    """
    Full pipeline: build team aggregates, add baseline (shrunk) ratings,
    then add SoS-adjusted rating_sos.
    """
    agg = build_team_aggregates(train)
    out = add_ratings(agg, alpha=alpha)
    out = add_sos_rating(out, train, sos_weight=sos_weight, sos_iters=sos_iters)
    return out
