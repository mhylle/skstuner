#!/usr/bin/env python3
"""Generate synthetic training data"""
import logging
import json
import sys
from pathlib import Path
import click
from skstuner.config import Config
from skstuner.data.sks_parser import SKSCode
from skstuner.data.synthetic_generator import SyntheticDataGenerator
from skstuner.data.checkpoint_manager import CheckpointManager
from skstuner.data.quality_validator import QualityValidator, calculate_diversity_metrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--codes-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/processed/sks_codes.json"),
    help="Input SKS codes JSON file",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=Path("data/synthetic/train_data.json"),
    help="Output dataset file",
)
@click.option(
    "--examples-per-code", type=int, default=10, help="Number of examples to generate per code"
)
@click.option(
    "--max-codes", type=int, default=None, help="Maximum number of codes to process (for testing)"
)
@click.option("--category", type=str, default=None, help="Filter by category (D, K, B, N, U, ZZ)")
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume from checkpoint if available (default: False)",
)
@click.option(
    "--checkpoint-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to checkpoint file (default: auto-generated based on output file)",
)
@click.option(
    "--checkpoint-interval",
    type=int,
    default=5,
    help="Save checkpoint every N codes (default: 5)",
)
@click.option(
    "--enable-quality-filter/--no-quality-filter",
    default=True,
    help="Enable quality validation and filtering (default: True)",
)
@click.option(
    "--quality-threshold",
    type=float,
    default=0.5,
    help="Minimum quality score to keep examples (0-1, default: 0.5)",
)
@click.option(
    "--show-quality-report/--no-quality-report",
    default=True,
    help="Show detailed quality report after generation (default: True)",
)
def main(
    codes_file: Path,
    output_file: Path,
    examples_per_code: int,
    max_codes: int,
    category: str,
    resume: bool,
    checkpoint_file: Path,
    checkpoint_interval: int,
    enable_quality_filter: bool,
    quality_threshold: float,
    show_quality_report: bool,
):
    """Generate synthetic training data for SKS codes"""
    logger.info("Starting synthetic data generation")

    # Load config
    config = Config()

    # Validate API key is set
    try:
        config.validate_api_key("anthropic_api_key")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Setup checkpoint file path
    if checkpoint_file is None:
        checkpoint_file = output_file.parent / f".{output_file.stem}_checkpoint.json"

    # Initialize checkpoint manager if resuming
    checkpoint_manager = None
    if resume:
        checkpoint_manager = CheckpointManager(checkpoint_file)
        if checkpoint_manager.load():
            logger.info("Resuming from existing checkpoint")
            stats = checkpoint_manager.get_statistics()
            logger.info(
                f"  Processed: {stats['processed_codes']} codes, "
                f"Failed: {stats['failed_codes']}, "
                f"Examples: {stats['total_examples']}"
            )
        else:
            logger.info("No checkpoint found, starting fresh")
    elif checkpoint_file.exists():
        # Not resuming but checkpoint exists - ask user what to do
        logger.warning(
            f"Checkpoint file exists at {checkpoint_file}. "
            f"Use --resume to continue from checkpoint, or delete it to start fresh."
        )
        if not click.confirm("Delete existing checkpoint and start fresh?", default=False):
            logger.info("Aborted by user")
            sys.exit(0)
        checkpoint_file.unlink()
        logger.info("Deleted existing checkpoint")

    # Create checkpoint manager for saving (even if not resuming)
    if checkpoint_manager is None:
        checkpoint_manager = CheckpointManager(checkpoint_file)

    # Load SKS codes
    logger.info(f"Loading codes from {codes_file}")
    with open(codes_file) as f:
        codes_data = json.load(f)

    # Convert to SKSCode objects
    codes = [
        SKSCode(
            code=c["code"],
            description=c["description"],
            category=c["category"],
            level=c["level"],
            parent_code=c.get("parent_code"),
        )
        for c in codes_data["codes"]
    ]

    # Filter by category if specified
    if category:
        codes = [c for c in codes if c.category == category]
        logger.info(f"Filtered to {len(codes)} codes in category {category}")

    # Limit codes if specified
    if max_codes:
        codes = codes[:max_codes]
        logger.info(f"Limited to {max_codes} codes for testing")

    logger.info(f"Generating {examples_per_code} examples for {len(codes)} codes")

    # Initialize quality validator if enabled
    quality_validator = None
    if enable_quality_filter:
        quality_validator = QualityValidator(quality_threshold=quality_threshold)
        logger.info(
            f"Quality filtering enabled with threshold: {quality_threshold} "
            f"(examples below this score will be rejected)"
        )

    # Generate dataset
    generator = SyntheticDataGenerator(api_key=config.anthropic_api_key)
    dataset = generator.generate_dataset(
        codes=codes,
        examples_per_code=examples_per_code,
        checkpoint_manager=checkpoint_manager,
        checkpoint_interval=checkpoint_interval,
        quality_validator=quality_validator,
    )

    # Save dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ Saved {len(dataset)} examples to {output_file}")

    # Print statistics
    categories = {}
    for example in dataset:
        cat = example["category"]
        categories[cat] = categories.get(cat, 0) + 1

    logger.info("\nDataset statistics:")
    logger.info(f"  Total examples: {len(dataset)}")
    logger.info(f"  Unique codes: {len(codes)}")
    logger.info(f"  Categories: {categories}")

    # Generate quality report if requested
    if show_quality_report and dataset:
        logger.info("\n" + "=" * 60)
        logger.info("QUALITY REPORT")
        logger.info("=" * 60)

        # Run quality validation on full dataset
        if quality_validator:
            logger.info("Running quality validation on generated dataset...")
            validation_stats = quality_validator.validate_dataset(dataset, show_progress=False)

            logger.info(f"\nQuality Validation Results:")
            logger.info(f"  Total examples: {validation_stats['total_examples']}")
            logger.info(
                f"  Passed: {validation_stats['passed']} ({validation_stats['pass_rate']:.1%})"
            )
            logger.info(
                f"  Failed: {validation_stats['failed']} ({1-validation_stats['pass_rate']:.1%})"
            )
            logger.info(f"  Average score: {validation_stats['average_score']:.3f}")

            if validation_stats["failed_examples"]:
                logger.info(f"\nSample of failed examples (first 5):")
                for i, failed in enumerate(validation_stats["failed_examples"][:5]):
                    logger.info(f"  {i+1}. Label: {failed['label']}, Score: {failed['score']:.3f}")
                    logger.info(f"     Issues: {', '.join(failed['issues'])}")

        # Calculate diversity metrics
        logger.info("\nCalculating diversity metrics...")
        diversity = calculate_diversity_metrics(dataset)

        logger.info(f"\nDiversity Metrics:")
        logger.info(f"  Unique texts: {diversity['unique_texts']} / {diversity['total_examples']}")
        logger.info(f"  Uniqueness ratio: {diversity['unique_ratio']:.1%}")
        logger.info(f"  Average text length: {diversity['avg_text_length']:.1f} chars")
        logger.info(f"  Vocabulary size: {diversity['vocabulary_size']} words")
        logger.info(
            f"  Vocabulary diversity: {diversity['vocabulary_diversity_ratio']:.3f} "
            f"(unique words / total words)"
        )

        if diversity.get("most_common_words"):
            logger.info(f"\n  Most common words (excluding stop words):")
            for word, count in diversity["most_common_words"][:10]:
                logger.info(f"    {word}: {count}")

        logger.info("\n" + "=" * 60)

    # Clean up checkpoint on successful completion
    if checkpoint_manager and click.confirm(
        "\nGeneration complete. Delete checkpoint file?", default=True
    ):
        checkpoint_manager.delete()


if __name__ == "__main__":
    main()
