#!/usr/bin/env python3
"""Generate synthetic training data"""
import logging
import json
from pathlib import Path
import click
from skstuner.config import Config
from skstuner.data.sks_parser import SKSCode
from skstuner.data.synthetic_generator import SyntheticDataGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.command()
@click.option('--codes-file', type=click.Path(exists=True, path_type=Path),
              default=Path("data/processed/sks_codes.json"),
              help='Input SKS codes JSON file')
@click.option('--output-file', type=click.Path(path_type=Path),
              default=Path("data/synthetic/train_data.json"),
              help='Output dataset file')
@click.option('--examples-per-code', type=int, default=10,
              help='Number of examples to generate per code')
@click.option('--max-codes', type=int, default=None,
              help='Maximum number of codes to process (for testing)')
@click.option('--category', type=str, default=None,
              help='Filter by category (D, K, B, N, U, ZZ)')
def main(codes_file: Path, output_file: Path, examples_per_code: int,
         max_codes: int, category: str):
    """Generate synthetic training data for SKS codes"""
    logger.info("Starting synthetic data generation")

    # Load config
    config = Config()

    if not config.anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY not set in environment")
        return

    # Load SKS codes
    logger.info(f"Loading codes from {codes_file}")
    with open(codes_file) as f:
        codes_data = json.load(f)

    # Convert to SKSCode objects
    codes = [
        SKSCode(
            code=c['code'],
            description=c['description'],
            category=c['category'],
            level=c['level'],
            parent_code=c.get('parent_code')
        )
        for c in codes_data['codes']
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

    # Generate dataset
    generator = SyntheticDataGenerator(api_key=config.anthropic_api_key)
    dataset = generator.generate_dataset(
        codes=codes,
        examples_per_code=examples_per_code
    )

    # Save dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ Saved {len(dataset)} examples to {output_file}")

    # Print statistics
    categories = {}
    for example in dataset:
        cat = example['category']
        categories[cat] = categories.get(cat, 0) + 1

    logger.info("\nDataset statistics:")
    logger.info(f"  Total examples: {len(dataset)}")
    logger.info(f"  Unique codes: {len(codes)}")
    logger.info(f"  Categories: {categories}")


if __name__ == "__main__":
    main()
