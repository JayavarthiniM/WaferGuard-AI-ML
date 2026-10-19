"""Lightweight, repeatable integrity checks for the Wafer-GUARD challenge data.

Run from the AI_ML folder (or pass the CSV path):
    python audit_checks.py [path/to/secom_telemetry_modified.csv]

These checks validate source-data assumptions and split boundaries. They do not
train a model and do not read the evaluator-only mask.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("secom_telemetry_modified.csv")

def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)
    print(f"PASS: {message}")

def main():
    df = pd.read_csv(DATA_PATH)
    check({"label", "timestamp"}.issubset(df.columns), "required label/timestamp columns exist")
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    check(len(sensor_cols) == 590, "dataset has 590 sensor columns")
    check(len(df) == 1567, "dataset has 1,567 rows")
    check(set(pd.to_numeric(df["label"], errors="raise").unique()) <= {-1, 1}, "labels use only -1 (pass) and 1 (defect)")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    check(df["timestamp"].is_monotonic_increasing, "chronological ordering is monotonic")

    n = len(df)
    a, b = int(n * 0.60), int(n * 0.80)
    check(0 < a < b < n, "chronological split boundaries are valid")
    check(df.loc[a - 1, "timestamp"] <= df.loc[a, "timestamp"] <= df.loc[b, "timestamp"],
          "train/validation/test periods remain in chronological order")

    raw = df[sensor_cols]
    legacy = raw["sensor_0"].eq(0.0)
    allowed_codes = {100.0, 200.0, 300.0}
    legacy_values = set(raw.loc[legacy, "sensor_1"].dropna().unique())
    check(int(legacy.sum()) == 146, "146 legacy rows are inferred from sensor_0 == 0.0")
    check(legacy_values <= allowed_codes, "legacy sensor_1 values are categorical firmware codes only")
    check(legacy_values == allowed_codes, "all three documented legacy firmware codes occur")

    prevalence = (df["label"].to_numpy() == 1).mean()
    check(abs(prevalence - 104 / 1567) < 1e-12, "natural defect prevalence matches 104/1,567")
    print(f"Dataset prevalence: {prevalence:.4%}")
    print(f"Chronological rows: train={a}, validation={b-a}, test={n-b}")
    print("NOTE: this script does not validate model quality or prove all upstream data is causal.")

if __name__ == "__main__":
    main()
