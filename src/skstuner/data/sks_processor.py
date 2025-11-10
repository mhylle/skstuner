"""Process and export SKS codes"""

from pathlib import Path
from typing import List, Dict
import json
import logging
from collections import defaultdict
from skstuner.data.sks_parser import SKSCode

logger = logging.getLogger(__name__)


class SKSProcessor:
    """Process SKS codes for training"""

    def __init__(self, codes: List[SKSCode]):
        self.codes = codes

    def export_json(self, output_path: Path):
        """
        Export codes to JSON format

        Args:
            output_path: Path to output JSON file
        """
        data = {
            "total_codes": len(self.codes),
            "codes": [
                {
                    "code": code.code,
                    "description": code.description,
                    "category": code.category,
                    "level": code.level,
                    "parent_code": code.parent_code,
                    "children": code.children,
                }
                for code in self.codes
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(self.codes)} codes to {output_path}")

    def export_taxonomy(self, output_path: Path):
        """
        Export label taxonomy for model training

        Args:
            output_path: Path to output JSON file
        """
        # Create label to ID mapping
        label2id = {
            code.code: idx for idx, code in enumerate(sorted(self.codes, key=lambda x: x.code))
        }
        id2label = {idx: code for code, idx in label2id.items()}

        # Create category to labels mapping
        category_labels = defaultdict(list)
        for code in self.codes:
            category_labels[code.category].append(code.code)

        # Create level to labels mapping
        level_labels = defaultdict(list)
        for code in self.codes:
            level_labels[code.level].append(code.code)

        taxonomy = {
            "num_labels": len(self.codes),
            "label2id": label2id,
            "id2label": id2label,
            "categories": dict(category_labels),
            "levels": {str(k): v for k, v in level_labels.items()},
            "descriptions": {code.code: code.description for code in self.codes},
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(taxonomy, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported taxonomy to {output_path}")

    def get_statistics(self) -> Dict:
        """
        Get statistics about the codes

        Returns:
            Dictionary with statistics
        """
        category_counts = defaultdict(int)
        level_counts = defaultdict(int)

        for code in self.codes:
            category_counts[code.category] += 1
            level_counts[code.level] += 1

        return {
            "total_codes": len(self.codes),
            "categories": dict(category_counts),
            "levels": dict(level_counts),
            "top_level_codes": len([c for c in self.codes if c.level == 1]),
        }

    def filter_by_category(self, category: str) -> List[SKSCode]:
        """
        Filter codes by category

        Args:
            category: Category to filter by (D, K, B, N, U, ZZ)

        Returns:
            List of codes in category
        """
        return [code for code in self.codes if code.category == category]

    def filter_by_level(self, level: int) -> List[SKSCode]:
        """
        Filter codes by hierarchy level

        Args:
            level: Hierarchy level to filter by

        Returns:
            List of codes at level
        """
        return [code for code in self.codes if code.level == level]
