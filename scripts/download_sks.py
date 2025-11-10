#!/usr/bin/env python3
"""Download SKS classification codes"""
import logging
from pathlib import Path
import click
from skstuner.data.sks_downloader import SKSDownloader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw"),
    help="Output directory for downloaded file",
)
@click.option("--force", is_flag=True, help="Force download even if file exists")
def main(output_dir: Path, force: bool):
    """Download SKS classification codes from Sundhedsdatastyrelsen"""
    logger.info("Starting SKS code download")

    downloader = SKSDownloader(output_dir=output_dir)
    output_file = downloader.download(force=force)

    logger.info(f"✓ SKS codes downloaded to {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
