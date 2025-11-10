#!/usr/bin/env python3
"""Validate quality of existing synthetic data"""
import logging
import json
import sys
from pathlib import Path
import click
from skstuner.data.quality_validator import (
    QualityValidator,
    calculate_diversity_metrics,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input dataset JSON file to validate",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional: Save filtered dataset to this file",
)
@click.option(
    "--quality-threshold",
    type=float,
    default=0.5,
    help="Minimum quality score to keep examples (0-1, default: 0.5)",
)
@click.option(
    "--filter/--no-filter",
    default=False,
    help="Remove low-quality examples (requires --output-file)",
)
@click.option(
    "--report-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional: Save detailed validation report to JSON file",
)
def main(
    input_file: Path,
    output_file: Path,
    quality_threshold: float,
    filter: bool,
    report_file: Path,
):
    """Validate and optionally filter synthetic training data"""
    logger.info(f"Loading dataset from {input_file}")

    # Load dataset
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    if not isinstance(dataset, list):
        logger.error("Dataset must be a JSON array of examples")
        sys.exit(1)

    logger.info(f"Loaded {len(dataset)} examples")

    # Initialize validator
    validator = QualityValidator(quality_threshold=quality_threshold)

    # Run validation
    logger.info("\n" + "=" * 60)
    logger.info("QUALITY VALIDATION")
    logger.info("=" * 60)

    validation_stats = validator.validate_dataset(dataset, show_progress=True)

    logger.info(f"\nValidation Results:")
    logger.info(f"  Total examples: {validation_stats['total_examples']}")
    logger.info(f"  Passed: {validation_stats['passed']} ({validation_stats['pass_rate']:.1%})")
    logger.info(
        f"  Failed: {validation_stats['failed']} ({1-validation_stats['pass_rate']:.1%})"
    )
    logger.info(f"  Average score: {validation_stats['average_score']:.3f}")

    if validation_stats["failed_examples"]:
        logger.info(f"\nSample of failed examples (first 10):")
        for i, failed in enumerate(validation_stats["failed_examples"][:10]):
            logger.info(f"  {i+1}. Label: {failed['label']}, Score: {failed['score']:.3f}")
            logger.info(f"     Issues: {', '.join(failed['issues'])}")
            logger.info(f"     Preview: {failed['text_preview']}")

    # Calculate diversity metrics
    logger.info("\n" + "=" * 60)
    logger.info("DIVERSITY METRICS")
    logger.info("=" * 60)

    diversity = calculate_diversity_metrics(dataset)

    logger.info(f"\nDiversity Results:")
    logger.info(f"  Total examples: {diversity['total_examples']}")
    logger.info(
        f"  Unique texts: {diversity['unique_texts']} ({diversity['unique_ratio']:.1%})"
    )
    logger.info(f"  Average text length: {diversity['avg_text_length']:.1f} chars")
    logger.info(f"  Length variance: {diversity['length_variance']:.1f}")
    logger.info(f"  Vocabulary size: {diversity['vocabulary_size']} words")
    logger.info(f"  Total words: {diversity['total_words']}")
    logger.info(
        f"  Vocabulary diversity ratio: {diversity['vocabulary_diversity_ratio']:.3f}"
    )

    if diversity.get("most_common_words"):
        logger.info(f"\n  Most common words (top 15):")
        for word, count in diversity["most_common_words"][:15]:
            logger.info(f"    {word}: {count}")

    # Category breakdown if available
    categories = {}
    for ex in dataset:
        cat = ex.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    if categories:
        logger.info(f"\n  Category distribution:")
        for cat, count in sorted(categories.items()):
            logger.info(f"    {cat}: {count} ({count/len(dataset):.1%})")

    # Filter dataset if requested
    if filter:
        if not output_file:
            logger.error("--filter requires --output-file to be specified")
            sys.exit(1)

        logger.info("\n" + "=" * 60)
        logger.info("FILTERING DATASET")
        logger.info("=" * 60)

        filtered_dataset = validator.filter_dataset(dataset, show_progress=True)

        # Save filtered dataset
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_dataset, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✓ Saved {len(filtered_dataset)} filtered examples to {output_file}")
        logger.info(
            f"  Removed: {len(dataset) - len(filtered_dataset)} examples "
            f"({(len(dataset) - len(filtered_dataset))/len(dataset):.1%})"
        )

    # Save detailed report if requested
    if report_file:
        report = {
            "input_file": str(input_file),
            "quality_threshold": quality_threshold,
            "validation": validation_stats,
            "diversity": diversity,
            "categories": categories,
        }

        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✓ Saved detailed report to {report_file}")

    logger.info("\n" + "=" * 60)
    logger.info("Validation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
