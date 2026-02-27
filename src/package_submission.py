"""
Package submission artifacts (Predictions.csv, Rankings.xlsx) into Submission.zip.
"""
import zipfile
from pathlib import Path


def create_submission_zip(
    predictions_path: Path,
    rankings_path: Path,
    zip_path: Path,
) -> None:
    """
    Create Submission.zip containing only Predictions.csv and Rankings.xlsx.
    Uses base names so the zip contains Predictions.csv and Rankings.xlsx at root.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_path, "Predictions.csv")
        zf.write(rankings_path, "Rankings.xlsx")
