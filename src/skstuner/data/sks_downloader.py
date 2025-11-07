"""SKS code downloader from Sundhedsdatastyrelsen"""
from pathlib import Path
import requests
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SKSDownloader:
    """Downloads SKS classification codes from official source"""

    SKS_FTP_BASE = "https://filer.sundhedsdata.dk/sks/data/skscomplete/"
    SKS_COMPLETE_FILE = "SKScomplete.txt"

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self, force: bool = False) -> Path:
        """
        Download SKScomplete.txt file

        Args:
            force: If True, download even if file exists

        Returns:
            Path to downloaded file
        """
        output_file = self.output_dir / self.SKS_COMPLETE_FILE

        if output_file.exists() and not force:
            logger.info(f"SKS file already exists at {output_file}")
            return output_file

        url = f"{self.SKS_FTP_BASE}{self.SKS_COMPLETE_FILE}"
        logger.info(f"Downloading SKS codes from {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download SKS codes: {e}")
            raise

        output_file.write_bytes(response.content)
        logger.info(f"Downloaded SKS codes to {output_file}")

        # Validate the file
        self.validate_file(output_file)

        return output_file

    def validate_file(self, file_path: Path):
        """
        Validate that the downloaded file has expected format

        Args:
            file_path: Path to file to validate

        Raises:
            ValueError: If file format is invalid
        """
        content = file_path.read_text(encoding='latin-1')
        lines = content.split('\n')

        if len(lines) < 10:
            raise ValueError("Invalid SKS file format: too few lines")

        # Check that lines have expected structure (17 fields separated by delimiters)
        # This is a basic validation - actual parsing will be more thorough
        first_line = lines[0]
        if len(first_line) < 50:
            raise ValueError("Invalid SKS file format: lines too short")

        logger.info(f"SKS file validated: {len(lines)} lines")
