# RPL Baseline Pipeline

Clean, minimal, extensible baseline for the **Rocketball Premier League (RPL)** contest. Ingests regular-season results and a derby/test file, computes team ratings, ranks all teams, and predicts derby match winners and margins.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place your data in `data/raw/`:

- **Train.csv** — Regular-season games (home/away teams and scores). Must have columns for home team, away team, home score, away score (names are inferred: e.g. `HomeTeam`/`home_team`, `HomePts`/`home_score`, etc.).
- **Derby/test file** — Matchups to predict (no scores). Use one of: `Derbies.csv`, `Test.csv`, or `Predictions.csv`. Must have two team columns (e.g. `Team1`/`Team2` or `TeamA`/`TeamB`). Optional: match ID (e.g. `GameID`/`MatchID`).

If required columns cannot be inferred, the pipeline raises clear errors listing expected name variants and the columns found.

## Run

From the project root:

```bash
python run_pipeline.py --data_dir data/raw
```

**Options:**

| Option        | Default              | Description                                      |
|---------------|----------------------|--------------------------------------------------|
| `--data_dir`  | `data/raw`           | Directory containing Train and derby/test file   |
| `--train`     | auto (`Train.csv`)   | Train filename                                   |
| `--derby`     | auto-detect          | Derby filename (Derbies.csv, Test.csv, Predictions.csv) |
| `--alpha`     | 5                    | Shrinkage for rating (higher = more shrink toward 0)    |
| `--out_dir`   | `data/outputs`      | Directory for timestamped outputs                |

Example with custom alpha:

```bash
python run_pipeline.py --data_dir data/raw --alpha 10
```

## Outputs

- **Timestamped** (in `data/outputs/`): `Predictions_YYYYMMDD_HHMMSS.csv`, `Rankings_YYYYMMDD_HHMMSS.xlsx`
- **Final** (project root): `Predictions.csv`, `Rankings.xlsx`
- **Submission**: `Submission.zip` in project root, containing only `Predictions.csv` and `Rankings.xlsx`

### Rankings.xlsx

| Column | Description        |
|--------|--------------------|
| Rank   | 1 = best … 165 = worst |
| Team   | Team name          |
| Rating | Baseline rating (shrunk avg margin) |

### Predictions.csv

| Column            | Description                          |
|-------------------|--------------------------------------|
| GameID            | Match identifier (if present in input) |
| Team1, Team2      | Matchup teams                        |
| PredictedWinner   | Predicted winner                     |
| PredictedMargin   | Predicted victory margin (non-negative) |
| Team1_WinMargin   | Margin from Team1’s perspective (signed) |

## Pipeline Overview

1. **Load** — `src/io_utils.py`: infer schema and load Train + derby CSVs.
2. **Features** — `src/feature_engineering.py`: optional train/derby feature hooks (baseline: pass-through).
3. **Rating** — `src/rating.py`: per-team aggregates (games, wins, avg margin, home/away splits) and baseline rating with shrinkage: `rating_shrunk = (games_played / (games_played + alpha)) * avg_margin`.
4. **Rank** — `src/rank.py`: rank all teams by `rating_shrunk` (1 = best).
5. **Predict** — `src/predict.py`: neutral-site derby prediction; predicted margin = rating(Team1) − rating(Team2); winner and margin derived from that (ties → Team1, margin 0).
6. **Package** — `src/package_submission.py`: build `Submission.zip`.

## Extending the Pipeline

### Feature engineering

- **`src/feature_engineering.py`**
  - `add_train_features(df)` — add columns to the training DataFrame (e.g. strength of schedule, rest days).
  - `add_derby_features(derby_df, team_stats, rating_col)` — add columns to the derby DataFrame (e.g. head-to-head, venue).

New features can be joined in `rating.py` (for team-level features) or in `predict.py` (for matchup-level features).

### Rating and prediction

- **`src/rating.py`** — Change `add_ratings()` or add new rating columns (e.g. Elo, strength-of-schedule adjusted).
- **`src/predict.py`** — Use extra features in `predict_derbies()` (e.g. home advantage if not neutral, or a simple model on margin).

Keep the same output schema (`Rankings.xlsx`: Rank, Team, Rating; `Predictions.csv`: GameID, Team1, Team2, PredictedWinner, PredictedMargin, Team1_WinMargin) so submission packaging stays unchanged.

## Next Steps

- **Validation**: Add a holdout or time-based split on `Train.csv`, score predicted margin (e.g. MAE) and winner accuracy.
- **Features**: Implement strength of schedule, recent form, or head-to-head in `feature_engineering.py` and wire them into rating/predict.
- **Rating**: Try alternative formulas (e.g. Bayesian shrinkage, Elo) in `rating.py` and compare on the same validation setup.
- **Submission**: Run the pipeline, then submit `Submission.zip` to the competition.
