"""Script to train SKS classification models."""

import argparse
import json
import logging
from pathlib import Path

from skstuner.config import Config
from skstuner.utils.logging_config import setup_logging
from skstuner.training.dataset import prepare_datasets
from skstuner.training.model import (
    load_model_config,
    create_model,
    save_model,
)
from skstuner.training.trainer import train_model
from skstuner.training.metrics import (
    compute_detailed_metrics,
    print_evaluation_summary,
    save_metrics,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SKS classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train XLM-RoBERTa on synthetic data
  python scripts/train_model.py \\
      --data-file data/synthetic/train_data.json \\
      --config models/configs/xlm_roberta_large.yaml \\
      --output-dir models/trained/xlm_roberta

  # Train Phi-3 with LoRA
  python scripts/train_model.py \\
      --data-file data/synthetic/train_data.json \\
      --config models/configs/phi3_mini.yaml \\
      --output-dir models/trained/phi3 \\
      --test-size 0.15 \\
      --val-size 0.1

  # Train Gemma with custom settings
  python scripts/train_model.py \\
      --data-file data/synthetic/train_data.json \\
      --config models/configs/gemma_7b.yaml \\
      --output-dir models/trained/gemma \\
      --max-length 256 \\
      --batch-size 2
        """,
    )

    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to synthetic training data JSON file",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model configuration YAML file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save trained model and outputs",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2)",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of train data for validation set (default: 0.1)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum sequence length (overrides config)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache downloaded models",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging(level=getattr(logging, args.log_level))

    logger.info("=" * 80)
    logger.info("SKS MODEL TRAINING")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading model config from {args.config}")
    model_config = load_model_config(Path(args.config))

    # Override config with command line arguments
    if args.max_length is not None:
        model_config["max_length"] = args.max_length
    if args.batch_size is not None:
        model_config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        model_config["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        model_config["num_epochs"] = args.num_epochs

    max_length = model_config.get("max_length", 512)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_output = output_dir / "training_config.json"
    with open(config_output, "w") as f:
        json.dump(
            {
                "data_file": args.data_file,
                "model_config": model_config,
                "test_size": args.test_size,
                "val_size": args.val_size,
                "max_length": max_length,
                "seed": args.seed,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved training configuration to {config_output}")

    # Step 1: Create model and tokenizer
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Creating Model and Tokenizer")
    logger.info("=" * 80)

    # For initial model creation, we'll use a temporary num_labels
    # This will be updated after we load the data
    temp_model, tokenizer = create_model(
        config=model_config,
        num_labels=100,  # Temporary value
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )

    # Step 2: Prepare datasets
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Preparing Datasets")
    logger.info("=" * 80)

    datasets, label_to_id, id_to_label = prepare_datasets(
        data_path=Path(args.data_file),
        tokenizer=tokenizer,
        max_length=max_length,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )

    num_labels = len(label_to_id)
    logger.info(f"Number of unique SKS codes: {num_labels}")

    # Update model config with actual number of labels
    model_config["num_labels"] = num_labels

    # Recreate model with correct number of labels
    logger.info("Recreating model with correct number of labels...")
    model, tokenizer = create_model(
        config=model_config,
        num_labels=num_labels,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )

    # Save label mappings
    label_mapping_path = output_dir / "label_mapping.json"
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_to_id": label_to_id,
                "id_to_label": {str(k): v for k, v in id_to_label.items()},
                "num_labels": num_labels,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Saved label mappings to {label_mapping_path}")

    # Step 3: Train model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Training Model")
    logger.info("=" * 80)

    trained_model, metrics = train_model(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        output_dir=output_dir / "checkpoints",
        config=model_config,
        num_labels=num_labels,
    )

    # Step 4: Detailed evaluation on test set
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Detailed Evaluation")
    logger.info("=" * 80)

    from transformers import Trainer, TrainingArguments

    # Create a temporary trainer for predictions
    eval_args = TrainingArguments(
        output_dir=str(output_dir / "temp"),
        per_device_eval_batch_size=model_config.get("batch_size", 16),
        remove_unused_columns=False,
    )

    eval_trainer = Trainer(
        model=trained_model,
        args=eval_args,
        tokenizer=tokenizer,
    )

    # Get predictions
    predictions = eval_trainer.predict(datasets["test"])

    # Compute detailed metrics
    detailed_metrics = compute_detailed_metrics(
        predictions=predictions.predictions,
        labels=predictions.label_ids,
        id_to_label=id_to_label,
        top_k=5,
    )

    # Print and save metrics
    print_evaluation_summary(detailed_metrics)
    save_metrics(detailed_metrics, output_dir / "test_metrics.json")

    # Step 5: Save final model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Saving Final Model")
    logger.info("=" * 80)

    final_model_dir = output_dir / "final_model"
    save_model(
        model=trained_model,
        tokenizer=tokenizer,
        output_dir=final_model_dir,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
    )

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Model saved to: {final_model_dir}")
    logger.info(f"Metrics saved to: {output_dir / 'test_metrics.json'}")
    logger.info(f"Test F1 Score: {detailed_metrics['weighted_f1']:.4f}")
    logger.info(f"Test Accuracy: {detailed_metrics['top1_accuracy']:.4f}")
    logger.info(f"Top-5 Accuracy: {detailed_metrics.get('top5_accuracy', 0):.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
