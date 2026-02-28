"""
RPL baseline pipeline configuration.
Central place for paths and model hyperparameters.
"""
from pathlib import Path

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_OUTPUTS = PROJECT_ROOT / "data" / "outputs"

# Default filenames (auto-detect if not found)
DEFAULT_TRAIN_FILE = "Train.csv"
DEFAULT_DERBY_CANDIDATES = ("Derbies.csv", "Test.csv", "Predictions.csv")

# Rating shrinkage: shrink toward 0 when games_played is small.
# rating_shrunk = (games_played / (games_played + alpha)) * avg_margin
ALPHA = 5

# Strength of Schedule (SoS) adjusted rating:
# rating_sos = rating_shrunk + sos_weight * opponent_strength_avg (iterative)
SOS_WEIGHT = 0.5
SOS_ITERS = 5

# Output filenames in project root (for submission)
OUTPUT_PREDICTIONS_CSV = "Predictions.csv"
OUTPUT_RANKINGS_XLSX = "Rankings.xlsx"
SUBMISSION_ZIP = "Submission.zip"
