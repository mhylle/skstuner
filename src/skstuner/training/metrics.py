"""Evaluation metrics for SKS classification."""

import logging
from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for SKS classification.

    Args:
        eval_pred: EvalPrediction object with predictions and labels

    Returns:
        Dictionary of metric names and values
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(predictions, axis=1)

    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    # Macro averages (unweighted)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }

    return metrics


def compute_detailed_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    id_to_label: Dict[int, str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Compute detailed evaluation metrics including top-k accuracy.

    Args:
        predictions: Model predictions (logits or probabilities) [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        id_to_label: Mapping from label IDs to SKS codes
        top_k: Number of top predictions to consider

    Returns:
        Dictionary with detailed metrics
    """
    # Top-1 accuracy
    top1_preds = np.argmax(predictions, axis=1)
    top1_accuracy = accuracy_score(labels, top1_preds)

    # Top-k accuracy
    top_k_preds = np.argsort(predictions, axis=1)[:, -top_k:]
    top_k_correct = np.array([label in top_k_preds[i] for i, label in enumerate(labels)])
    top_k_accuracy = np.mean(top_k_correct)

    # Precision, recall, F1 (weighted and macro)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, top1_preds, average="weighted", zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, top1_preds, average="macro", zero_division=0
    )

    # Per-class metrics
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        labels, top1_preds, average=None, zero_division=0
    )

    # Get unique label names
    target_names = [id_to_label.get(i, f"Label_{i}") for i in range(len(class_precision))]

    # Generate classification report
    report = classification_report(
        labels,
        top1_preds,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    # Find best and worst performing classes
    class_f1_dict = {
        id_to_label.get(i, f"Label_{i}"): f1
        for i, f1 in enumerate(class_f1)
        if class_support[i] > 0  # Only consider classes with samples
    }

    if class_f1_dict:
        best_classes = sorted(class_f1_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        worst_classes = sorted(class_f1_dict.items(), key=lambda x: x[1])[:10]
    else:
        best_classes = []
        worst_classes = []

    metrics = {
        "top1_accuracy": float(top1_accuracy),
        f"top{top_k}_accuracy": float(top_k_accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "num_classes": len(class_precision),
        "total_samples": len(labels),
        "classification_report": report,
        "best_performing_classes": best_classes,
        "worst_performing_classes": worst_classes,
    }

    return metrics


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted summary of evaluation metrics.

    Args:
        metrics: Dictionary of metrics from compute_detailed_metrics
    """
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)

    # Overall metrics
    logger.info(f"\nOverall Performance:")
    logger.info(f"  Top-1 Accuracy:     {metrics['top1_accuracy']:.4f}")

    if "top5_accuracy" in metrics:
        logger.info(f"  Top-5 Accuracy:     {metrics['top5_accuracy']:.4f}")

    logger.info(f"\nWeighted Metrics (by class frequency):")
    logger.info(f"  Precision:          {metrics['weighted_precision']:.4f}")
    logger.info(f"  Recall:             {metrics['weighted_recall']:.4f}")
    logger.info(f"  F1-Score:           {metrics['weighted_f1']:.4f}")

    logger.info(f"\nMacro Metrics (unweighted):")
    logger.info(f"  Precision:          {metrics['macro_precision']:.4f}")
    logger.info(f"  Recall:             {metrics['macro_recall']:.4f}")
    logger.info(f"  F1-Score:           {metrics['macro_f1']:.4f}")

    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total Samples:      {metrics['total_samples']}")
    logger.info(f"  Number of Classes:  {metrics['num_classes']}")

    # Best performing classes
    if metrics.get("best_performing_classes"):
        logger.info(f"\nTop 10 Best Performing Classes:")
        for i, (class_name, score) in enumerate(metrics["best_performing_classes"], 1):
            logger.info(f"  {i:2d}. {class_name:30s} F1: {score:.4f}")

    # Worst performing classes
    if metrics.get("worst_performing_classes"):
        logger.info(f"\nTop 10 Worst Performing Classes:")
        for i, (class_name, score) in enumerate(metrics["worst_performing_classes"], 1):
            logger.info(f"  {i:2d}. {class_name:30s} F1: {score:.4f}")

    logger.info("=" * 80)


def save_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Save evaluation metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        output_path: Path to save metrics JSON
    """
    import json
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"Metrics saved to {output_path}")
