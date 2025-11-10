"""Tests for evaluation metrics."""

import numpy as np
import pytest

from skstuner.training.metrics import (
    compute_metrics,
    compute_detailed_metrics,
)


@pytest.fixture
def sample_predictions():
    """Create sample predictions and labels."""
    # 5 samples, 3 classes
    predictions = np.array(
        [
            [0.8, 0.1, 0.1],  # Correct: predicts 0, label is 0
            [0.2, 0.7, 0.1],  # Correct: predicts 1, label is 1
            [0.1, 0.1, 0.8],  # Correct: predicts 2, label is 2
            [0.3, 0.4, 0.3],  # Wrong: predicts 1, label is 0
            [0.6, 0.3, 0.1],  # Correct: predicts 0, label is 0
        ]
    )
    labels = np.array([0, 1, 2, 0, 0])
    return predictions, labels


@pytest.fixture
def id_to_label():
    """Create sample label mapping."""
    return {0: "DE11", 1: "DJ45", 2: "DI21"}


def test_compute_metrics(sample_predictions):
    """Test basic metrics computation."""
    predictions, labels = sample_predictions

    # Create EvalPrediction-like object
    class EvalPred:
        def __init__(self, predictions, label_ids):
            self.predictions = predictions
            self.label_ids = label_ids

    eval_pred = EvalPred(predictions, labels)
    metrics = compute_metrics(eval_pred)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "macro_f1" in metrics

    # Check accuracy is correct (4 out of 5)
    assert metrics["accuracy"] == 0.8


def test_compute_detailed_metrics(sample_predictions, id_to_label):
    """Test detailed metrics computation."""
    predictions, labels = sample_predictions

    metrics = compute_detailed_metrics(
        predictions=predictions, labels=labels, id_to_label=id_to_label, top_k=2
    )

    assert "top1_accuracy" in metrics
    assert "top2_accuracy" in metrics
    assert "weighted_precision" in metrics
    assert "weighted_recall" in metrics
    assert "weighted_f1" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "macro_f1" in metrics
    assert "num_classes" in metrics
    assert "total_samples" in metrics
    assert "classification_report" in metrics
    assert "best_performing_classes" in metrics
    assert "worst_performing_classes" in metrics

    # Check top-1 accuracy
    assert metrics["top1_accuracy"] == 0.8

    # Check top-2 accuracy (should be higher)
    assert metrics["top2_accuracy"] >= metrics["top1_accuracy"]

    # Check sample counts
    assert metrics["total_samples"] == 5
    assert metrics["num_classes"] == 3


def test_top_k_accuracy(id_to_label):
    """Test top-k accuracy computation."""
    # Create predictions where top-1 is wrong but top-3 includes correct label
    predictions = np.array(
        [
            [0.2, 0.5, 0.3],  # Top-1: 1, Top-2: [1,2], Label: 0 - wrong
            [0.3, 0.4, 0.3],  # Top-1: 1, Top-2: [1,0], Label: 1 - correct
        ]
    )
    labels = np.array([0, 1])

    metrics = compute_detailed_metrics(
        predictions=predictions, labels=labels, id_to_label=id_to_label, top_k=2
    )

    # Top-1 accuracy should be 0.5 (1 out of 2)
    assert metrics["top1_accuracy"] == 0.5

    # Top-2 accuracy should be 1.0 (both correct)
    assert metrics["top2_accuracy"] == 1.0


def test_perfect_predictions():
    """Test metrics with perfect predictions."""
    predictions = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    id_to_label = {0: "A", 1: "B"}

    metrics = compute_detailed_metrics(
        predictions=predictions, labels=labels, id_to_label=id_to_label, top_k=1
    )

    # All metrics should be 1.0 for perfect predictions
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
