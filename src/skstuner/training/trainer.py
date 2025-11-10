"""Training utilities for SKS classification models."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from datasets import DatasetDict

from skstuner.training.metrics import compute_metrics

logger = logging.getLogger(__name__)


class SKSTrainer:
    """Trainer for SKS classification models."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset,
        eval_dataset,
        output_dir: Path,
        config: Dict[str, Any],
        num_labels: int,
    ):
        """
        Initialize SKS trainer.

        Args:
            model: Model to train
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory for saving checkpoints and logs
            config: Training configuration
            num_labels: Number of classification labels
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.config = config
        self.num_labels = num_labels

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup training arguments
        self.training_args = self._create_training_arguments()

        # Create trainer
        self.trainer = self._create_trainer()

    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments from config."""
        logger.info("Creating training arguments...")

        # Extract config values
        batch_size = self.config.get("batch_size", 16)
        num_epochs = self.config.get("num_epochs", 3)
        learning_rate = self.config.get("learning_rate", 2e-5)
        warmup_steps = self.config.get("warmup_steps", 500)
        weight_decay = self.config.get("weight_decay", 0.01)
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)

        # Determine if we can use GPU
        device_count = torch.cuda.device_count()
        use_gpu = torch.cuda.is_available()

        logger.info(f"Training configuration:")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Weight decay: {weight_decay}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  GPU available: {use_gpu}")
        logger.info(f"  GPU count: {device_count}")

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",  # Disable W&B by default
            remove_unused_columns=False,
            push_to_hub=False,
            fp16=use_gpu,  # Use mixed precision if GPU available
            dataloader_num_workers=4 if use_gpu else 0,
            seed=42,
        )

        return training_args

    def _create_trainer(self) -> Trainer:
        """Create HuggingFace Trainer."""
        logger.info("Creating Trainer...")

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        return trainer

    def train(self) -> Dict[str, Any]:
        """
        Train the model.

        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Evaluation samples: {len(self.eval_dataset)}")

        # Train
        train_result = self.trainer.train()

        # Save final model
        logger.info(f"Saving final model to {self.output_dir}")
        self.trainer.save_model(str(self.output_dir / "final_model"))

        # Get metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(self.train_dataset)

        # Log metrics
        logger.info("Training completed!")
        logger.info(f"Training metrics: {metrics}")

        return metrics

    def evaluate(self, dataset=None) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            dataset: Optional dataset to evaluate on (uses eval_dataset if not provided)

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model...")

        if dataset is None:
            dataset = self.eval_dataset

        metrics = self.trainer.evaluate(dataset)

        logger.info(f"Evaluation metrics: {metrics}")

        return metrics

    def predict(self, dataset):
        """
        Make predictions on a dataset.

        Args:
            dataset: Dataset to predict on

        Returns:
            PredictionOutput with predictions, labels, and metrics
        """
        logger.info(f"Making predictions on {len(dataset)} samples...")

        predictions = self.trainer.predict(dataset)

        return predictions


def train_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    datasets: DatasetDict,
    output_dir: Path,
    config: Dict[str, Any],
    num_labels: int,
) -> tuple[PreTrainedModel, Dict[str, Any]]:
    """
    Train a model for SKS classification.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        datasets: DatasetDict with train/validation/test splits
        output_dir: Directory for saving outputs
        config: Training configuration
        num_labels: Number of classification labels

    Returns:
        Tuple of (trained_model, metrics)
    """
    # Create trainer
    trainer = SKSTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        output_dir=output_dir,
        config=config,
        num_labels=num_labels,
    )

    # Train
    train_metrics = trainer.train()

    # Evaluate on validation set
    val_metrics = trainer.evaluate(datasets["validation"])

    # Evaluate on test set
    test_metrics = trainer.evaluate(datasets["test"])

    # Combine all metrics
    all_metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    logger.info("=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Validation F1: {val_metrics.get('eval_f1', 0):.4f}")
    logger.info(f"Test F1: {test_metrics.get('eval_f1', 0):.4f}")
    logger.info(f"Test Accuracy: {test_metrics.get('eval_accuracy', 0):.4f}")
    logger.info("=" * 80)

    return trainer.model, all_metrics
