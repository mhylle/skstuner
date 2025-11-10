"""Parser for SKS classification codes"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Constants for SKS file format
MIN_LINE_LENGTH = 131
CODE_LENGTH_LEVEL_1 = 3
CODE_LENGTH_LEVEL_2 = 4
CODE_LENGTH_LEVEL_3 = 5
VALID_CATEGORIES = {"D", "K", "B", "N", "U", "ZZ"}


@dataclass
class SKSCode:
    """Represents a single SKS code"""

    code: str
    description: str
    category: str  # D, K, B, N, U, ZZ
    level: int
    parent_code: Optional[str] = None
    children: List[str] = None

    def __post_init__(self) -> None:
        """Initialize default values and validate data"""
        if self.children is None:
            self.children = []

        # Validate category
        if self.category not in VALID_CATEGORIES:
            logger.warning(f"Unexpected category '{self.category}' for code {self.code}")


class SKSParser:
    """Parser for SKS classification files"""

    # Field positions in fixed-width format (0-indexed)
    # Based on SKS documentation
    FIELD_POSITIONS = {
        "code": (0, 5),
        "description": (5, 70),
        "valid_from": (70, 80),
        "valid_to": (80, 90),
        "category": (130, 131),
    }

    def parse_line(self, line: str) -> Optional[SKSCode]:
        """
        Parse a single line from SKS file

        Args:
            line: Fixed-width format line

        Returns:
            SKSCode object or None if invalid
        """
        if len(line) < MIN_LINE_LENGTH:
            return None

        try:
            # Extract fields using fixed positions
            code = line[self.FIELD_POSITIONS["code"][0] : self.FIELD_POSITIONS["code"][1]].strip()
            description = line[
                self.FIELD_POSITIONS["description"][0] : self.FIELD_POSITIONS["description"][1]
            ].strip()
            category = line[
                self.FIELD_POSITIONS["category"][0] : self.FIELD_POSITIONS["category"][1]
            ].strip()

            if not code or not category:
                return None

            # Determine hierarchy level based on code structure
            level = self._determine_level(code)
            parent_code = self._determine_parent(code)

            return SKSCode(
                code=code,
                description=description,
                category=category,
                level=level,
                parent_code=parent_code,
            )
        except (IndexError, ValueError) as e:
            logger.warning(f"Failed to parse line (length: {len(line)}): {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing line: {e}", exc_info=True)
            return None

    def _determine_level(self, code: str) -> int:
        """
        Determine hierarchy level from code structure

        Args:
            code: SKS code string

        Returns:
            Hierarchy level (1-4)

        Notes:
            Level 1: D50 (3 chars)
            Level 2: D500, D50A (4 chars)
            Level 3: D500A (5 chars)
            Level 4: longer codes
        """
        code_clean = code.strip()
        code_len = len(code_clean)

        if code_len <= CODE_LENGTH_LEVEL_1:
            return 1
        elif code_len == CODE_LENGTH_LEVEL_2:
            return 2
        elif code_len == CODE_LENGTH_LEVEL_3:
            return 3
        else:
            return 4

    def _determine_parent(self, code: str) -> Optional[str]:
        """
        Determine parent code from code structure

        Args:
            code: SKS code string

        Returns:
            Parent code string or None if top-level
        """
        code_clean = code.strip()

        if len(code_clean) <= CODE_LENGTH_LEVEL_1:
            return None  # Top level

        # Parent is code with last character removed
        return code_clean[:-1]

    def parse_file(self, file_path: Path) -> List[SKSCode]:
        """
        Parse complete SKS file

        Args:
            file_path: Path to SKS file

        Returns:
            List of SKSCode objects
        """
        logger.info(f"Parsing SKS file: {file_path}")

        content = file_path.read_text(encoding="latin-1")
        lines = content.split("\n")

        codes = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue

            code = self.parse_line(line)
            if code:
                codes.append(code)

        logger.info(f"Parsed {len(codes)} SKS codes")
        return codes

    def build_hierarchy(self, codes: List[SKSCode]) -> Dict[str, Dict]:
        """
        Build hierarchical structure from flat list of codes

        Args:
            codes: List of SKS codes

        Returns:
            Nested dictionary representing hierarchy
        """
        # Create code lookup
        code_dict = {code.code: code for code in codes}

        # Build hierarchy
        hierarchy = {}

        for code in codes:
            if code.parent_code and code.parent_code in code_dict:
                # Add to parent's children
                parent = code_dict[code.parent_code]
                if code.code not in parent.children:
                    parent.children.append(code.code)

            # Add to hierarchy root if top-level or parent missing
            if not code.parent_code or code.parent_code not in code_dict:
                hierarchy[code.code] = {
                    "description": code.description,
                    "category": code.category,
                    "level": code.level,
                    "children": {},
                }

        # Build nested structure for codes with parents
        def add_children(parent_code: str, hierarchy_node: Dict):
            if parent_code not in code_dict:
                return

            parent = code_dict[parent_code]
            for child_code in parent.children:
                if child_code in code_dict:
                    child = code_dict[child_code]
                    hierarchy_node["children"][child_code] = {
                        "description": child.description,
                        "category": child.category,
                        "level": child.level,
                        "children": {},
                    }
                    add_children(child_code, hierarchy_node["children"][child_code])

        # Populate children recursively
        for root_code in list(hierarchy.keys()):
            add_children(root_code, hierarchy[root_code])

        return hierarchy
