"""SKS code downloader from Sundhedsdatastyrelsen"""

from pathlib import Path
import requests
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Constants for validation
MIN_FILE_LINES = 10
MIN_LINE_LENGTH = 50
DEFAULT_TIMEOUT = 30


class SKSDownloader:
    """Downloads SKS classification codes from official source"""

    SKS_FTP_BASE = "https://filer.sundhedsdata.dk/sks/data/skscomplete/"
    SKS_COMPLETE_FILE = "SKScomplete.txt"

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the SKS downloader

        Args:
            output_dir: Directory where to save downloaded files. Defaults to data/raw
        """
        self.output_dir = output_dir or Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self, force: bool = False) -> Path:
        """
        Download SKScomplete.txt file

        Args:
            force: If True, download even if file exists

        Returns:
            Path to downloaded file

        Raises:
            requests.exceptions.RequestException: If download fails
            ValueError: If downloaded file is invalid
        """
        output_file = self.output_dir / self.SKS_COMPLETE_FILE

        if output_file.exists() and not force:
            logger.info(f"SKS file already exists at {output_file}")
            # Validate existing file
            self.validate_file(output_file)
            return output_file

        url = f"{self.SKS_FTP_BASE}{self.SKS_COMPLETE_FILE}"
        logger.info(f"Downloading SKS codes from {url}")

        try:
            response = requests.get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            logger.error(f"Download timed out after {DEFAULT_TIMEOUT}s: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during download: {e}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error during download: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download SKS codes: {e}")
            raise

        if not response.content:
            raise ValueError("Downloaded file is empty")

        output_file.write_bytes(response.content)
        logger.info(f"Downloaded SKS codes to {output_file} ({len(response.content)} bytes)")

        # Validate the file
        self.validate_file(output_file)

        return output_file

    def validate_file(self, file_path: Path) -> None:
        """
        Validate that the downloaded file has expected format

        Args:
            file_path: Path to file to validate

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.stat().st_size == 0:
            raise ValueError(f"File is empty: {file_path}")

        try:
            content = file_path.read_text(encoding="latin-1")
        except UnicodeDecodeError as e:
            raise ValueError(f"File encoding error (expected latin-1): {e}") from e

        lines = content.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        if len(non_empty_lines) < MIN_FILE_LINES:
            raise ValueError(
                f"Invalid SKS file format: too few lines. "
                f"Expected at least {MIN_FILE_LINES}, got {len(non_empty_lines)}"
            )

        # Check that first non-empty line has expected structure
        first_line = non_empty_lines[0]
        if len(first_line) < MIN_LINE_LENGTH:
            raise ValueError(
                f"Invalid SKS file format: first line too short. "
                f"Expected at least {MIN_LINE_LENGTH} chars, got {len(first_line)}"
            )

        logger.info(
            f"SKS file validated: {len(lines)} total lines, {len(non_empty_lines)} non-empty"
        )
