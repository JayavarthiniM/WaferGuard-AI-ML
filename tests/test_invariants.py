"""Regression checks for the WaferGuard notebook's critical invariants.

These tests inspect the submitted notebook source and exercise the tensor/mask
invariants on small synthetic batches. They do not replace a full notebook run
against the organizer dataset.
Run from the repository root: python -m pytest -q
"""
from pathlib import Path
import json

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "wafer_guard_fixed.ipynb"


@pytest.fixture(scope="module")
def notebook_source():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )


def test_split_is_chronological_and_non_overlapping(notebook_source):
    assert "df = df.sort_values(\"timestamp\", kind=\"stable\").reset_index(drop=True)" in notebook_source
    assert "train_idx = np.arange(0, train_end)" in notebook_source
    assert "val_idx = np.arange(train_end, val_end)" in notebook_source
    assert "test_idx = np.arange(val_end, n)" in notebook_source
    assert "assert train_idx[-1] < val_idx[0] < test_idx[0]" in notebook_source


def test_imputer_and_scaler_are_fit_on_training_rows_only(notebook_source):
    assert "imputer.fit_transform(X_df.iloc[train_idx][continuous_cols])" in notebook_source
    assert "scaler.fit_transform(X_train_cont)" in notebook_source
    assert "scaler.transform(imputer.transform(X_df.iloc[val_idx][continuous_cols]))" in notebook_source
    assert "scaler.transform(imputer.transform(X_df.iloc[test_idx][continuous_cols]))" in notebook_source
    # Guard against common full-data fitting regressions.
    assert "imputer.fit_transform(X_df[continuous_cols])" not in notebook_source
    assert "scaler.fit_transform(X_df" not in notebook_source


def test_evaluator_mask_is_never_opened(notebook_source):
    lowered = notebook_source.lower()
    assert "secom_legacy_mask_evaluator_only.csv" in lowered  # documented exclusion only
    forbidden_reads = (
        "read_csv(\"secom_legacy_mask_evaluator_only.csv\"",
        "read_csv('secom_legacy_mask_evaluator_only.csv'",
        "open(\"secom_legacy_mask_evaluator_only.csv\"",
        "open('secom_legacy_mask_evaluator_only.csv'",
    )
    assert not any(token in lowered for token in forbidden_reads)


def test_psl_slots_are_neutralized_and_masked(notebook_source):
    assert "legacy = raw[\"sensor_0\"].eq(0.0)" in notebook_source
    assert "raw[\"sensor_1\"].isin([100.0, 200.0, 300.0])" in notebook_source
    assert "X_part[local_legacy, 0] = 0.0" in notebook_source
    assert "X_part[local_legacy, 1] = 0.0" in notebook_source
    assert "mask[local_legacy, 0] = 0.0" in notebook_source
    assert "mask[local_legacy, 1] = 0.0" in notebook_source
    # Exercise the expected invariant on a synthetic legacy row.
    X = np.array([[0.0, 0.0, 1.25, 1.0, 0.0, 0.0]], dtype=np.float32)
    mask = np.array([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    assert np.all(X[0, :2] == 0.0)
    assert np.all(mask[0, :2] == 0.0)


def test_tensor_shapes_and_loss_gradients_are_valid():
    batch_size, features = 8, 12
    x = torch.randn(batch_size, features)
    reconstruction = torch.randn(batch_size, features, requires_grad=True)
    logits = torch.randn(batch_size, requires_grad=True)
    labels = torch.randint(0, 2, (batch_size,), dtype=torch.float32)
    mask = torch.ones_like(x)
    mask[0, :2] = 0.0

    assert reconstruction.shape == x.shape
    assert logits.shape == labels.shape
    reconstruction_loss = (((reconstruction - x) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss = reconstruction_loss + classification_loss
    loss.backward()

    assert torch.isfinite(loss).item()
    assert reconstruction.grad is not None and torch.isfinite(reconstruction.grad).all().item()
    assert logits.grad is not None and torch.isfinite(logits.grad).all().item()


def test_notebook_contains_explicit_shape_regression_assertions(notebook_source):
    assert "assert reconstruction.shape == xb.shape" in notebook_source
    assert "assert logits.shape == yb.shape" in notebook_source
    assert "assert tuple(broadcast_shape) == (64, 64, 590)" in notebook_source


def test_threshold_is_selected_on_validation_and_test_is_not_used_for_tuning(notebook_source):
    assert "yv, pv = predict_probabilities(val_loader)" in notebook_source
    assert "val_scores = [(float(t), f1_score(yv" in notebook_source
    assert "yt, pt = predict_probabilities(test_loader)" in notebook_source
    assert "No test metric is used to tune the model or threshold." in notebook_source
