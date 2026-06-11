"""Project configuration and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Samarpan_BloodBank_SyntheticDataset_V2.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORTS_DIR = PROJECT_ROOT / "reports"

RETENTION_WINDOW_DAYS = 180
CHURN_WINDOW_DAYS = 365
RANDOM_STATE = 42
TEST_SIZE = 0.2

AGE_BINS = [0, 25, 35, 45, 60, 120]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-60", "60+"]

for directory in (FIGURES_DIR, MODELS_DIR, METRICS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
