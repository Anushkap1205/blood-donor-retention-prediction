"""Load Samarpan blood bank tables."""

from __future__ import annotations

import pandas as pd

from src.config import DATA_PATH


def load_all_tables(path: str | None = None) -> dict[str, pd.DataFrame]:
    """Load all Excel sheets into a dictionary of DataFrames."""
    file_path = path or str(DATA_PATH)
    sheets = {
        "donors": "Donor_Master",
        "donations": "Donation_Register",
        "patients": "Patient_Master",
        "hospitals": "Hospital_Master",
        "issues": "Blood_Issue_Register",
        "inventory": "Monthly_Inventory",
    }
    return {key: pd.read_excel(file_path, sheet_name=sheet) for key, sheet in sheets.items()}


def load_donor_donation_tables(path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load donor master and donation register tables."""
    tables = load_all_tables(path)
    return tables["donors"], tables["donations"]
