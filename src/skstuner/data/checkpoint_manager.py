"""Checkpoint management for synthetic data generation"""

from pathlib import Path
from typing import List, Dict, Set, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints for resumable data generation"""

    def __init__(self, checkpoint_file: Path):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_file: Path to checkpoint file
        """
        self.checkpoint_file = checkpoint_file
        self.processed_codes: Set[str] = set()
        self.failed_codes: Set[str] = set()
        self.dataset: List[Dict] = []
        self.metadata: Dict = {}

    def load(self) -> bool:
        """
        Load checkpoint from file

        Returns:
            True if checkpoint was loaded, False if file doesn't exist
        """
        if not self.checkpoint_file.exists():
            logger.info(f"No checkpoint found at {self.checkpoint_file}")
            return False

        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            self.processed_codes = set(checkpoint_data.get("processed_codes", []))
            self.failed_codes = set(checkpoint_data.get("failed_codes", []))
            self.dataset = checkpoint_data.get("dataset", [])
            self.metadata = checkpoint_data.get("metadata", {})

            logger.info(
                f"Loaded checkpoint: {len(self.processed_codes)} codes processed, "
                f"{len(self.failed_codes)} failed, {len(self.dataset)} examples"
            )
            return True

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def save(self) -> None:
        """Save checkpoint to file"""
        checkpoint_data = {
            "processed_codes": list(self.processed_codes),
            "failed_codes": list(self.failed_codes),
            "dataset": self.dataset,
            "metadata": {
                **self.metadata,
                "last_updated": datetime.now().isoformat(),
            },
        }

        # Create parent directory if it doesn't exist
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first, then rename for atomicity
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

            # Atomic rename
            temp_file.replace(self.checkpoint_file)

            logger.debug(f"Checkpoint saved: {len(self.processed_codes)} codes processed")

        except IOError as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def is_code_processed(self, code: str) -> bool:
        """Check if a code has been processed"""
        return code in self.processed_codes

    def mark_code_processed(self, code: str) -> None:
        """Mark a code as successfully processed"""
        self.processed_codes.add(code)

    def mark_code_failed(self, code: str) -> None:
        """Mark a code as failed"""
        self.failed_codes.add(code)

    def add_examples(self, examples: List[Dict]) -> None:
        """Add examples to dataset"""
        self.dataset.extend(examples)

    def get_remaining_codes(self, all_codes: List[str]) -> List[str]:
        """
        Get list of codes that haven't been processed yet

        Args:
            all_codes: List of all code strings

        Returns:
            List of unprocessed code strings
        """
        return [code for code in all_codes if code not in self.processed_codes]

    def delete(self) -> None:
        """Delete checkpoint file"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info(f"Deleted checkpoint file: {self.checkpoint_file}")

    def get_statistics(self) -> Dict:
        """Get checkpoint statistics"""
        return {
            "processed_codes": len(self.processed_codes),
            "failed_codes": len(self.failed_codes),
            "total_examples": len(self.dataset),
            "last_updated": self.metadata.get("last_updated", "Never"),
        }
