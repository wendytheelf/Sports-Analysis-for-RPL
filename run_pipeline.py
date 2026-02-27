#!/usr/bin/env python3
"""
End-to-end RPL baseline pipeline: load data -> rate teams -> rank -> predict derbies -> save outputs -> package submission.
"""
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import (
    ALPHA,
    DATA_OUTPUTS,
    DATA_RAW,
    DEFAULT_DERBY_CANDIDATES,
    DEFAULT_TRAIN_FILE,
    OUTPUT_PREDICTIONS_CSV,
    OUTPUT_RANKINGS_XLSX,
    PROJECT_ROOT,
    SUBMISSION_ZIP,
)
from src.feature_engineering import add_train_features
from src.io_utils import detect_derby_file, load_derby_csv, load_train_csv
from src.package_submission import create_submission_zip
from src.predict import build_predictions_csv, predict_derbies, save_predictions_csv
from src.rank import build_rankings, save_rankings
from src.rating import compute_ratings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RPL baseline pipeline")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DATA_RAW,
        help="Directory containing Train.csv and derby/test file",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Train filename (default: auto-detect Train.csv)",
    )
    parser.add_argument(
        "--derby",
        type=str,
        default=None,
        help="Derby/test filename (default: auto-detect from Derbies.csv, Test.csv, Predictions.csv)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="Shrinkage alpha for rating (default: 5)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DATA_OUTPUTS,
        help="Directory for timestamped outputs",
    )
    return parser.parse_args()


def run_pipeline(
    data_dir: Path,
    train_file: Optional[str],
    derby_file: Optional[str],
    alpha: float,
    out_dir: Path,
) -> None:
    data_dir = data_dir.resolve()
    out_dir = out_dir.resolve()

    # 1) Load train
    train_path = data_dir / (train_file or DEFAULT_TRAIN_FILE)
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    train = load_train_csv(train_path)
    train = add_train_features(train)

    # 2) Load derby
    if derby_file:
        derby_path = data_dir / derby_file
        if not derby_path.exists():
            raise FileNotFoundError(f"Derby file not found: {derby_path}")
    else:
        derby_path = detect_derby_file(data_dir)
        if derby_path is None:
            raise FileNotFoundError(
                f"No derby/test file found in {data_dir}. "
                f"Expected one of: {DEFAULT_DERBY_CANDIDATES}"
            )
    derby = load_derby_csv(derby_path)

    # 3) Compute team ratings
    team_stats = compute_ratings(train, alpha=alpha)

    # 4) Rank all teams
    rankings = build_rankings(team_stats, rating_col="rating_shrunk")

    # 5) Predict derbies
    predictions = predict_derbies(derby, team_stats, rating_col="rating_shrunk")
    predictions = build_predictions_csv(predictions)

    # 6) Save to out_dir with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_ts_path = out_dir / f"Predictions_{ts}.csv"
    rank_ts_path = out_dir / f"Rankings_{ts}.xlsx"
    save_predictions_csv(predictions, pred_ts_path)
    save_rankings(rankings, rank_ts_path)

    # 7) Save final names in project root (for submission)
    pred_root = PROJECT_ROOT / OUTPUT_PREDICTIONS_CSV
    rank_root = PROJECT_ROOT / OUTPUT_RANKINGS_XLSX
    save_predictions_csv(predictions, pred_root)
    save_rankings(rankings, rank_root)

    # 8) Create Submission.zip in project root
    zip_path = PROJECT_ROOT / SUBMISSION_ZIP
    create_submission_zip(pred_root, rank_root, zip_path)

    print(f"Train: {train_path} ({len(train)} games)")
    print(f"Derby: {derby_path} ({len(derby)} matches)")
    print(f"Teams ranked: {len(rankings)}")
    print(f"Timestamped outputs: {pred_ts_path}, {rank_ts_path}")
    print(f"Root outputs: {pred_root}, {rank_root}")
    print(f"Submission: {zip_path}")


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        train_file=args.train,
        derby_file=args.derby,
        alpha=args.alpha,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
