#!/usr/bin/env python3
"""Process SKS codes and export to JSON"""
import logging
from pathlib import Path
import click
from skstuner.data.sks_parser import SKSParser
from skstuner.data.sks_processor import SKSProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.command()
@click.option('--input-file', type=click.Path(exists=True, path_type=Path),
              default=Path("data/raw/SKScomplete.txt"),
              help='Input SKS file')
@click.option('--output-dir', type=click.Path(path_type=Path),
              default=Path("data/processed"),
              help='Output directory')
def main(input_file: Path, output_dir: Path):
    """Process SKS codes and export to JSON formats"""
    logger.info(f"Processing SKS codes from {input_file}")

    # Parse codes
    parser = SKSParser()
    codes = parser.parse_file(input_file)

    # Process and export
    processor = SKSProcessor(codes=codes)

    # Export full codes
    codes_file = output_dir / "sks_codes.json"
    processor.export_json(codes_file)
    logger.info(f"✓ Exported codes to {codes_file}")

    # Export taxonomy
    taxonomy_file = output_dir / "sks_taxonomy.json"
    processor.export_taxonomy(taxonomy_file)
    logger.info(f"✓ Exported taxonomy to {taxonomy_file}")

    # Print statistics
    stats = processor.get_statistics()
    logger.info("\nStatistics:")
    logger.info(f"  Total codes: {stats['total_codes']}")
    logger.info(f"  Top-level codes: {stats['top_level_codes']}")
    logger.info(f"  Categories: {stats['categories']}")
    logger.info(f"  Levels: {stats['levels']}")


if __name__ == "__main__":
    main()
