"""
Ridge regression model for predicting game margin.
Builds match-level features from team stats and trains sklearn Ridge(alpha=1.0)
to reduce multicollinearity among features.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from src.io_utils import DERBY_TEAM1, DERBY_TEAM2, DERBY_TEAM1_ID, DERBY_TEAM2_ID
from src.io_utils import TRAIN_HOME_TEAM, TRAIN_AWAY_TEAM, TRAIN_HOME_SCORE, TRAIN_AWAY_SCORE
from src.io_utils import TRAIN_HOME_ID, TRAIN_AWAY_ID


def build_training_dataset(
    train_df: pd.DataFrame,
    team_stats: pd.DataFrame,
    rating_col: str = "rating_shrunk",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build matchup-level feature matrix X and target y from regular-season games.

    Uses only rating_diff (home rating - away rating) from the chosen rating_col
    (e.g. rating_shrunk or rating_sos). Target y = HomePts - AwayPts (signed margin).

    Returns:
        X: DataFrame with single column rating_diff
        y: Series of signed margin (home - away)
    """
    use_id = TRAIN_HOME_ID in train_df.columns and TRAIN_AWAY_ID in train_df.columns
    if use_id and "team_id" in team_stats.columns:
        home_stats = team_stats.set_index("team_id").add_suffix("_home")
        away_stats = team_stats.set_index("team_id").add_suffix("_away")
        train = train_df.merge(
            home_stats,
            left_on=TRAIN_HOME_ID,
            right_index=True,
            how="left",
        )
        train = train.merge(
            away_stats,
            left_on=TRAIN_AWAY_ID,
            right_index=True,
            how="left",
        )
    else:
        home_stats = team_stats.set_index("team").add_suffix("_home")
        away_stats = team_stats.set_index("team").add_suffix("_away")
        train = train_df.merge(
            home_stats,
            left_on=TRAIN_HOME_TEAM,
            right_index=True,
            how="left",
        )
        train = train.merge(
            away_stats,
            left_on=TRAIN_AWAY_TEAM,
            right_index=True,
            how="left",
        )

    train = train.copy()
    home_col = f"{rating_col}_home"
    away_col = f"{rating_col}_away"
    train["rating_diff"] = train[home_col] - train[away_col]

    feature_cols = ["rating_diff"]
    X = train[feature_cols].copy()
    X = X.fillna(0)

    y = train[TRAIN_HOME_SCORE] - train[TRAIN_AWAY_SCORE]
    return X, y


def train_linear_model(X: pd.DataFrame, y: pd.Series):
    """
    Fit sklearn Ridge(alpha=1.0) on (X, y). Default fit_intercept=True.
    Print training RMSE, R², and coefficient table (rating_diff + intercept).
    """
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X, y)
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print(f"Training RMSE: {rmse:.4f}")
    print(f"Training R²: {r2:.4f}")

    coef_df = pd.DataFrame({"feature": X.columns, "coef": model.coef_})
    print("\nCoefficient table:")
    print(coef_df.to_string(index=False))
    print(f"Intercept: {model.intercept_:.4f}")
    return model


def build_derby_features(
    derby_df: pd.DataFrame,
    team_stats: pd.DataFrame,
    rating_col: str = "rating_shrunk",
) -> pd.DataFrame:
    """
    Build feature matrix for derby matches. Single feature rating_diff (team1 - team2)
    from the chosen rating_col. Joins team_stats by team_id when present, else by team name.
    """
    use_id = DERBY_TEAM1_ID in derby_df.columns and DERBY_TEAM2_ID in derby_df.columns
    if use_id and "team_id" in team_stats.columns:
        team1_stats = team_stats.set_index("team_id").add_suffix("_1")
        team2_stats = team_stats.set_index("team_id").add_suffix("_2")
        derby = derby_df.merge(
            team1_stats,
            left_on=DERBY_TEAM1_ID,
            right_index=True,
            how="left",
        )
        derby = derby.merge(
            team2_stats,
            left_on=DERBY_TEAM2_ID,
            right_index=True,
            how="left",
        )
    else:
        team1_stats = team_stats.set_index("team").add_suffix("_1")
        team2_stats = team_stats.set_index("team").add_suffix("_2")
        derby = derby_df.merge(
            team1_stats,
            left_on=DERBY_TEAM1,
            right_index=True,
            how="left",
        )
        derby = derby.merge(
            team2_stats,
            left_on=DERBY_TEAM2,
            right_index=True,
            how="left",
        )

    derby = derby.copy()
    col1 = f"{rating_col}_1"
    col2 = f"{rating_col}_2"
    derby["rating_diff"] = derby[col1] - derby[col2]

    feature_cols = ["rating_diff"]
    X_derby = derby[feature_cols].copy()
    X_derby = X_derby.fillna(0)
    return X_derby


def predict_derbies_ml(model, derby_df: pd.DataFrame, X_derby: pd.DataFrame) -> pd.DataFrame:
    """
    Predict derby margins using the linear model. Returns full predictions DataFrame with
    PredictedWinner, PredictedMargin, Team1_WinMargin (signed, from Team1 perspective).
    """
    predicted_margin_signed = model.predict(X_derby)
    out = derby_df.copy()
    out["PredictedWinner"] = out[DERBY_TEAM1].where(
        predicted_margin_signed >= 0,
        out[DERBY_TEAM2],
    )
    out["PredictedMargin"] = np.abs(predicted_margin_signed).round(2)
    out["Team1_WinMargin"] = np.round(predicted_margin_signed, 2)
    return out
