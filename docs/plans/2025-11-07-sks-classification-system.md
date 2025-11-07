# SKS Classification System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete system to classify Danish clinical text into SKS (Sundhedsvæsenets Klassifikations System) codes using fine-tuned small LLMs with synthetic training data.

**Architecture:** Multi-task hierarchical classification using model-agnostic training pipeline. Supports multiple LLM backends (XLM-RoBERTa, Gemma, Phi-3) with easy model switching. Synthetic data generation via Claude/GPT-4. FastAPI inference service with comprehensive evaluation metrics.

**Tech Stack:** Python 3.10+, PyTorch 2.x, Transformers, PEFT (LoRA), FastAPI, Weights & Biases, Anthropic/OpenAI APIs

---

## Phase 1: Project Foundation

### Task 1: Project Structure and Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `.gitignore`
- Create: `src/skstuner/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create project structure**

Create directory structure:
```bash
mkdir -p src/skstuner/{data,models,training,evaluation,api,utils}
mkdir -p tests/{data,models,training,evaluation,api}
mkdir -p data/{raw,processed,synthetic}
mkdir -p models/configs
mkdir -p notebooks
mkdir -p scripts
touch src/skstuner/__init__.py
touch tests/__init__.py
```

**Step 2: Write pyproject.toml**

Create `pyproject.toml`:
```toml
[tool.poetry]
name = "skstuner"
version = "0.1.0"
description = "SKS code classification using fine-tuned LLMs"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
packages = [{include = "skstuner", from = "src"}]

[tool.poetry.dependencies]
python = "^3.10"
torch = "^2.1.0"
transformers = "^4.36.0"
peft = "^0.7.0"
datasets = "^2.16.0"
pandas = "^2.1.0"
numpy = "^1.24.0"
scikit-learn = "^1.3.0"
fastapi = "^0.109.0"
uvicorn = {extras = ["standard"], version = "^0.27.0"}
pydantic = "^2.5.0"
python-dotenv = "^1.0.0"
anthropic = "^0.25.0"
openai = "^1.10.0"
jinja2 = "^3.1.0"
wandb = "^0.16.0"
click = "^8.1.0"
tqdm = "^4.66.0"
requests = "^2.31.0"
httpx = "^0.26.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
black = "^24.0.0"
ruff = "^0.1.0"
ipython = "^8.12.0"
jupyter = "^1.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 100
target-version = ['py310']

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
```

**Step 3: Write .gitignore**

Create `.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Data
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
data/synthetic/*
!data/synthetic/.gitkeep

# Models
models/checkpoints/
models/saved/
*.pt
*.pth
*.bin
*.safetensors

# Logs
logs/
*.log
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Environment
.env
.env.local

# Jupyter
.ipynb_checkpoints/
*.ipynb
```

**Step 4: Write README**

Create `README.md`:
```markdown
# SKS Tuner

Fine-tuned LLM system for classifying Danish clinical text into SKS codes.

## Features

- Multi-task hierarchical classification
- Model-agnostic architecture (XLM-RoBERTa, Gemma, Phi-3)
- Synthetic data generation via LLM
- Comprehensive evaluation metrics
- FastAPI inference service

## Setup

```bash
# Install dependencies
poetry install

# Download SKS codes
python scripts/download_sks.py

# Generate synthetic data
python scripts/generate_synthetic_data.py

# Train model
python scripts/train.py --model xlm-roberta-large

# Run API
uvicorn src.skstuner.api.main:app --reload
```

## Project Structure

```
skstuner/
├── src/skstuner/
│   ├── data/           # Data processing
│   ├── models/         # Model architectures
│   ├── training/       # Training loops
│   ├── evaluation/     # Metrics and evaluation
│   ├── api/           # FastAPI service
│   └── utils/         # Utilities
├── tests/             # Test suite
├── data/              # Data storage
├── models/            # Model configs and checkpoints
├── scripts/           # CLI scripts
└── notebooks/         # Jupyter notebooks
```
```

**Step 5: Create placeholder files**

```bash
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/synthetic/.gitkeep
```

**Step 6: Initialize git and commit**

```bash
git init
git add .
git commit -m "feat: initial project structure and dependencies"
```

Expected: Clean commit with project structure

---

### Task 2: Configuration System

**Files:**
- Create: `src/skstuner/config.py`
- Create: `tests/test_config.py`
- Create: `.env.example`
- Create: `models/configs/xlm_roberta_large.yaml`
- Create: `models/configs/gemma_7b.yaml`

**Step 1: Write config test**

Create `tests/test_config.py`:
```python
import pytest
from pathlib import Path
from skstuner.config import Config, ModelConfig


def test_config_loads_from_env(tmp_path, monkeypatch):
    """Test that config loads from environment variables"""
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=test_key\nWANDB_PROJECT=test_project")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key")
    monkeypatch.setenv("WANDB_PROJECT", "test_project")

    config = Config()
    assert config.anthropic_api_key == "test_key"
    assert config.wandb_project == "test_project"


def test_model_config_loads_from_yaml(tmp_path):
    """Test loading model config from YAML"""
    config_content = """
model_name: xlm-roberta-large
model_type: encoder
num_labels: 1000
hidden_size: 1024
learning_rate: 2e-5
batch_size: 16
"""
    config_file = tmp_path / "test_model.yaml"
    config_file.write_text(config_content)

    model_config = ModelConfig.from_yaml(config_file)
    assert model_config.model_name == "xlm-roberta-large"
    assert model_config.learning_rate == 2e-5
    assert model_config.batch_size == 16


def test_config_paths_exist():
    """Test that config creates necessary paths"""
    config = Config()
    assert config.data_dir.exists()
    assert config.models_dir.exists()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: FAIL - Module 'skstuner.config' not found

**Step 3: Implement Config class**

Create `src/skstuner/config.py`:
```python
"""Configuration management for SKS Tuner"""
from pathlib import Path
from typing import Optional
import os
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Main application configuration"""

    # API Keys
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    wandb_api_key: str = field(default_factory=lambda: os.getenv("WANDB_API_KEY", ""))

    # Project settings
    wandb_project: str = field(default_factory=lambda: os.getenv("WANDB_PROJECT", "skstuner"))
    wandb_entity: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_ENTITY"))

    # Paths
    root_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.root_dir / "data"
        self.models_dir = self.root_dir / "models"
        self.logs_dir = self.root_dir / "logs"

        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "synthetic").mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Model-specific configuration"""

    model_name: str
    model_type: str  # "encoder" or "decoder"
    num_labels: int
    hidden_size: int
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_length: int = 512
    gradient_accumulation_steps: int = 1
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelConfig":
        """Load model config from YAML file"""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Path):
        """Save model config to YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: PASS - All config tests pass

**Step 5: Create example .env**

Create `.env.example`:
```bash
# API Keys
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
WANDB_API_KEY=your_wandb_key_here

# Weights & Biases
WANDB_PROJECT=skstuner
WANDB_ENTITY=your_wandb_entity

# Model settings
DEFAULT_MODEL=xlm-roberta-large
```

**Step 6: Create model config files**

Create `models/configs/xlm_roberta_large.yaml`:
```yaml
model_name: xlm-roberta-large
model_type: encoder
num_labels: 10000  # Will be updated after SKS parsing
hidden_size: 1024
learning_rate: 2.0e-05
batch_size: 16
num_epochs: 5
warmup_steps: 1000
weight_decay: 0.01
max_length: 512
gradient_accumulation_steps: 2
use_lora: false
```

Create `models/configs/gemma_7b.yaml`:
```yaml
model_name: google/gemma-7b
model_type: decoder
num_labels: 10000  # Will be updated after SKS parsing
hidden_size: 3072
learning_rate: 1.0e-04
batch_size: 4
num_epochs: 3
warmup_steps: 500
weight_decay: 0.01
max_length: 512
gradient_accumulation_steps: 8
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
```

Create `models/configs/phi3_mini.yaml`:
```yaml
model_name: microsoft/Phi-3-mini-4k-instruct
model_type: decoder
num_labels: 10000
hidden_size: 3072
learning_rate: 1.0e-04
batch_size: 8
num_epochs: 3
warmup_steps: 500
weight_decay: 0.01
max_length: 512
gradient_accumulation_steps: 4
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
```

**Step 7: Commit**

```bash
git add .
git commit -m "feat: add configuration system with model configs"
```

Expected: Clean commit

---

## Phase 2: SKS Data Processing

### Task 3: SKS Code Downloader

**Files:**
- Create: `src/skstuner/data/sks_downloader.py`
- Create: `tests/data/test_sks_downloader.py`
- Create: `scripts/download_sks.py`

**Step 1: Write downloader test**

Create `tests/data/test_sks_downloader.py`:
```python
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from skstuner.data.sks_downloader import SKSDownloader


def test_sks_downloader_downloads_file(tmp_path):
    """Test that SKS downloader downloads the file"""
    downloader = SKSDownloader(output_dir=tmp_path)

    # Mock the HTTP request
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.content = b"Test SKS data"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        output_file = downloader.download()

        assert output_file.exists()
        assert output_file.read_bytes() == b"Test SKS data"


def test_sks_downloader_validates_file_format(tmp_path):
    """Test that downloader validates SKS file format"""
    test_file = tmp_path / "SKScomplete.txt"
    test_file.write_text("Invalid format")

    downloader = SKSDownloader(output_dir=tmp_path)

    with pytest.raises(ValueError, match="Invalid SKS file format"):
        downloader.validate_file(test_file)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/data/test_sks_downloader.py -v
```

Expected: FAIL - Module not found

**Step 3: Implement SKS downloader**

Create `src/skstuner/data/sks_downloader.py`:
```python
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

        response = requests.get(url, timeout=30)
        response.raise_for_status()

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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/data/test_sks_downloader.py -v
```

Expected: PASS - All downloader tests pass

**Step 5: Create CLI script**

Create `scripts/download_sks.py`:
```python
#!/usr/bin/env python3
"""Download SKS classification codes"""
import logging
from pathlib import Path
import click
from skstuner.data.sks_downloader import SKSDownloader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.command()
@click.option('--output-dir', type=click.Path(path_type=Path), default=Path("data/raw"),
              help='Output directory for downloaded file')
@click.option('--force', is_flag=True, help='Force download even if file exists')
def main(output_dir: Path, force: bool):
    """Download SKS classification codes from Sundhedsdatastyrelsen"""
    logger.info("Starting SKS code download")

    downloader = SKSDownloader(output_dir=output_dir)
    output_file = downloader.download(force=force)

    logger.info(f"✓ SKS codes downloaded to {output_file}")
    logger.info(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
```

Make it executable:
```bash
chmod +x scripts/download_sks.py
```

**Step 6: Test the script manually**

```bash
python scripts/download_sks.py --help
```

Expected: Help message displays correctly

**Step 7: Commit**

```bash
git add .
git commit -m "feat: add SKS code downloader"
```

Expected: Clean commit

---

### Task 4: SKS Code Parser

**Files:**
- Create: `src/skstuner/data/sks_parser.py`
- Create: `tests/data/test_sks_parser.py`
- Create: `tests/data/fixtures/sample_sks.txt`

**Step 1: Write parser test**

Create `tests/data/fixtures/sample_sks.txt`:
```
D50  Jernmangelanæmi                                                  2018-01-01          9999-12-312018-01-01          9999-12-31D     001
D500 Jernmangelanæmi efter blødning                                   2018-01-01          9999-12-312018-01-01          9999-12-31D     002
D501 Sideropenisk dysfagi                                             2018-01-01          9999-12-312018-01-01          9999-12-31D     003
D50A Andre jernmangelanæmier                                          2018-01-01          9999-12-312018-01-01          9999-12-31D     004
```

Create `tests/data/test_sks_parser.py`:
```python
import pytest
from pathlib import Path
from skstuner.data.sks_parser import SKSParser, SKSCode


@pytest.fixture
def sample_sks_file(tmp_path):
    """Create a sample SKS file for testing"""
    sample_file = tmp_path / "sample_sks.txt"
    sample_content = """D50  Jernmangelanæmi                                                  2018-01-01          9999-12-312018-01-01          9999-12-31D     001
D500 Jernmangelanæmi efter blødning                                   2018-01-01          9999-12-312018-01-01          9999-12-31D     002                          """
    sample_file.write_text(sample_content, encoding='latin-1')
    return sample_file


def test_parse_sks_line():
    """Test parsing a single SKS line"""
    line = "D50  Jernmangelanæmi                                                  2018-01-01          9999-12-312018-01-01          9999-12-31D     001                          "

    parser = SKSParser()
    code = parser.parse_line(line)

    assert code.code == "D50"
    assert code.description == "Jernmangelanæmi"
    assert code.category == "D"
    assert code.level == 1


def test_parse_file_returns_codes(sample_sks_file):
    """Test parsing complete file"""
    parser = SKSParser()
    codes = parser.parse_file(sample_sks_file)

    assert len(codes) == 2
    assert codes[0].code == "D50"
    assert codes[1].code == "D500"


def test_build_hierarchy():
    """Test building code hierarchy"""
    codes = [
        SKSCode(code="D50", description="Jernmangelanæmi", category="D", level=1),
        SKSCode(code="D500", description="Jernmangelanæmi efter blødning", category="D", level=2),
        SKSCode(code="D501", description="Sideropenisk dysfagi", category="D", level=2),
    ]

    parser = SKSParser()
    hierarchy = parser.build_hierarchy(codes)

    assert "D50" in hierarchy
    assert len(hierarchy["D50"]["children"]) == 2
    assert "D500" in hierarchy["D50"]["children"]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/data/test_sks_parser.py -v
```

Expected: FAIL - Module not found

**Step 3: Implement SKS parser**

Create `src/skstuner/data/sks_parser.py`:
```python
"""Parser for SKS classification codes"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class SKSCode:
    """Represents a single SKS code"""
    code: str
    description: str
    category: str  # D, K, B, N, U, ZZ
    level: int
    parent_code: Optional[str] = None
    children: List[str] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class SKSParser:
    """Parser for SKS classification files"""

    # Field positions in fixed-width format (0-indexed)
    # Based on SKS documentation
    FIELD_POSITIONS = {
        'code': (0, 5),
        'description': (5, 70),
        'valid_from': (70, 80),
        'valid_to': (80, 90),
        'category': (110, 111),
    }

    def parse_line(self, line: str) -> Optional[SKSCode]:
        """
        Parse a single line from SKS file

        Args:
            line: Fixed-width format line

        Returns:
            SKSCode object or None if invalid
        """
        if len(line) < 120:
            return None

        try:
            # Extract fields using fixed positions
            code = line[self.FIELD_POSITIONS['code'][0]:self.FIELD_POSITIONS['code'][1]].strip()
            description = line[self.FIELD_POSITIONS['description'][0]:self.FIELD_POSITIONS['description'][1]].strip()
            category = line[self.FIELD_POSITIONS['category'][0]:self.FIELD_POSITIONS['category'][1]].strip()

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
                parent_code=parent_code
            )
        except Exception as e:
            logger.warning(f"Failed to parse line: {e}")
            return None

    def _determine_level(self, code: str) -> int:
        """Determine hierarchy level from code structure"""
        # Level 1: D50
        # Level 2: D500, D50A
        # Level 3: D500A
        # etc.

        code_clean = code.strip()

        # Count transitions between digit/letter
        transitions = 0
        prev_char_type = None

        for char in code_clean:
            if char == ' ':
                continue
            char_type = 'digit' if char.isdigit() else 'letter'
            if prev_char_type and char_type != prev_char_type:
                transitions += 1
            prev_char_type = char_type

        # Simple heuristic: level based on code length
        non_space_len = len(code_clean.replace(' ', ''))
        if non_space_len <= 2:
            return 1
        elif non_space_len == 3:
            return 2
        elif non_space_len == 4:
            return 3
        else:
            return 4

    def _determine_parent(self, code: str) -> Optional[str]:
        """Determine parent code from code structure"""
        code_clean = code.strip()

        if len(code_clean) <= 2:
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

        content = file_path.read_text(encoding='latin-1')
        lines = content.split('\n')

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
                    'description': code.description,
                    'category': code.category,
                    'level': code.level,
                    'children': {}
                }

        # Build nested structure for codes with parents
        def add_children(parent_code: str, hierarchy_node: Dict):
            if parent_code not in code_dict:
                return

            parent = code_dict[parent_code]
            for child_code in parent.children:
                if child_code in code_dict:
                    child = code_dict[child_code]
                    hierarchy_node['children'][child_code] = {
                        'description': child.description,
                        'category': child.category,
                        'level': child.level,
                        'children': {}
                    }
                    add_children(child_code, hierarchy_node['children'][child_code])

        # Populate children recursively
        for root_code in list(hierarchy.keys()):
            add_children(root_code, hierarchy[root_code])

        return hierarchy
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/data/test_sks_parser.py -v
```

Expected: PASS - All parser tests pass

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add SKS code parser with hierarchy building"
```

Expected: Clean commit

---

### Task 5: SKS Data Processor and Exporter

**Files:**
- Create: `src/skstuner/data/sks_processor.py`
- Create: `tests/data/test_sks_processor.py`
- Create: `scripts/process_sks.py`

**Step 1: Write processor test**

Create `tests/data/test_sks_processor.py`:
```python
import pytest
import json
from pathlib import Path
from skstuner.data.sks_processor import SKSProcessor
from skstuner.data.sks_parser import SKSCode


@pytest.fixture
def sample_codes():
    return [
        SKSCode(code="D50", description="Jernmangelanæmi", category="D", level=1),
        SKSCode(code="D500", description="Jernmangelanæmi efter blødning", category="D", level=2, parent_code="D50"),
        SKSCode(code="K01", description="Test procedure", category="K", level=1),
    ]


def test_processor_exports_json(tmp_path, sample_codes):
    """Test that processor exports codes to JSON"""
    processor = SKSProcessor(codes=sample_codes)
    output_file = tmp_path / "sks_codes.json"

    processor.export_json(output_file)

    assert output_file.exists()

    with open(output_file) as f:
        data = json.load(f)

    assert len(data['codes']) == 3
    assert data['total_codes'] == 3


def test_processor_gets_statistics(sample_codes):
    """Test getting statistics from codes"""
    processor = SKSProcessor(codes=sample_codes)
    stats = processor.get_statistics()

    assert stats['total_codes'] == 3
    assert stats['categories']['D'] == 2
    assert stats['categories']['K'] == 1
    assert stats['levels'][1] == 2
    assert stats['levels'][2] == 1


def test_processor_filters_by_category(sample_codes):
    """Test filtering codes by category"""
    processor = SKSProcessor(codes=sample_codes)
    d_codes = processor.filter_by_category('D')

    assert len(d_codes) == 2
    assert all(code.category == 'D' for code in d_codes)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/data/test_sks_processor.py -v
```

Expected: FAIL - Module not found

**Step 3: Implement processor**

Create `src/skstuner/data/sks_processor.py`:
```python
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
        self.code_dict = {code.code: code for code in codes}

    def export_json(self, output_path: Path):
        """
        Export codes to JSON format

        Args:
            output_path: Path to output JSON file
        """
        data = {
            'total_codes': len(self.codes),
            'codes': [
                {
                    'code': code.code,
                    'description': code.description,
                    'category': code.category,
                    'level': code.level,
                    'parent_code': code.parent_code,
                    'children': code.children
                }
                for code in self.codes
            ]
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported {len(self.codes)} codes to {output_path}")

    def export_taxonomy(self, output_path: Path):
        """
        Export label taxonomy for model training

        Args:
            output_path: Path to output JSON file
        """
        # Create label to ID mapping
        label2id = {code.code: idx for idx, code in enumerate(sorted(self.codes, key=lambda x: x.code))}
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
            'num_labels': len(self.codes),
            'label2id': label2id,
            'id2label': id2label,
            'categories': dict(category_labels),
            'levels': {str(k): v for k, v in level_labels.items()},
            'descriptions': {code.code: code.description for code in self.codes}
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
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
            'total_codes': len(self.codes),
            'categories': dict(category_counts),
            'levels': dict(level_counts),
            'top_level_codes': len([c for c in self.codes if c.level == 1])
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
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/data/test_sks_processor.py -v
```

Expected: PASS - All processor tests pass

**Step 5: Create processing script**

Create `scripts/process_sks.py`:
```python
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
```

Make it executable:
```bash
chmod +x scripts/process_sks.py
```

**Step 6: Commit**

```bash
git add .
git commit -m "feat: add SKS processor and export scripts"
```

Expected: Clean commit

---

## Phase 3: Synthetic Data Generation

### Task 6: Prompt Templates for Data Generation

**Files:**
- Create: `src/skstuner/data/prompt_templates.py`
- Create: `tests/data/test_prompt_templates.py`
- Create: `src/skstuner/data/templates/clinical_note.jinja2`

**Step 1: Write prompt template test**

Create `tests/data/test_prompt_templates.py`:
```python
import pytest
from skstuner.data.prompt_templates import PromptTemplateManager
from skstuner.data.sks_parser import SKSCode


def test_template_manager_renders_prompt():
    """Test that template manager renders prompts correctly"""
    manager = PromptTemplateManager()

    code = SKSCode(
        code="D50",
        description="Jernmangelanæmi",
        category="D",
        level=1
    )

    prompt = manager.render_clinical_note_prompt(code, num_examples=5)

    assert "D50" in prompt
    assert "Jernmangelanæmi" in prompt
    assert "5" in prompt or "fem" in prompt.lower()


def test_template_variations_specified():
    """Test that template includes variation instructions"""
    manager = PromptTemplateManager()

    code = SKSCode(code="D50", description="Jernmangelanæmi", category="D", level=1)
    prompt = manager.render_clinical_note_prompt(code, num_examples=3)

    # Should include instructions for variations
    assert any(word in prompt.lower() for word in ['variation', 'different', 'diverse'])
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/data/test_prompt_templates.py -v
```

Expected: FAIL - Module not found

**Step 3: Create Jinja2 template**

Create `src/skstuner/data/templates/clinical_note.jinja2`:
```jinja2
You are a Danish medical professional creating synthetic clinical notes for training a classification model.

**Task:** Generate {{ num_examples }} realistic Danish clinical notes that would be classified with SKS code {{ code.code }}.

**SKS Code Information:**
- Code: {{ code.code }}
- Description: {{ code.description }}
- Category: {{ code.category }}
{% if code.parent_code %}
- Parent code: {{ code.parent_code }}
{% endif %}

**Requirements:**

1. **Language:** All notes must be in Danish
2. **Variations:** Create diverse examples with different:
   - Note lengths (short: 2-3 sentences, medium: 1 paragraph, long: 2-3 paragraphs)
   - Note styles (emergency admission, outpatient visit, surgical report, progress note)
   - Patient demographics (age, gender when relevant)
   - Symptom presentations and severity levels
   - Medical terminology density (some technical, some plain language)

3. **Realism:** Notes should be realistic and could plausibly appear in Danish healthcare records:
   - Include relevant medical terminology in Danish
   - Include typical abbreviations used in Danish hospitals
   - Reflect real clinical reasoning and documentation patterns
   - Include both objective findings and subjective symptoms when appropriate

4. **Format:** Return ONLY the clinical notes, one per line, in this exact format:
   ```
   NOTE_1: [First clinical note text]
   NOTE_2: [Second clinical note text]
   NOTE_3: [Third clinical note text]
   ...
   ```

5. **Quality:** Each note should clearly relate to {{ code.description }} but express it in different ways. Avoid repetitive phrasing across notes.

Generate {{ num_examples }} varied clinical notes now:
```

**Step 4: Implement template manager**

Create `src/skstuner/data/prompt_templates.py`:
```python
"""Prompt templates for synthetic data generation"""
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template
from skstuner.data.sks_parser import SKSCode


class PromptTemplateManager:
    """Manages Jinja2 templates for LLM prompts"""

    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))

    def render_clinical_note_prompt(self, code: SKSCode, num_examples: int = 10) -> str:
        """
        Render prompt for generating clinical notes

        Args:
            code: SKS code to generate notes for
            num_examples: Number of examples to generate

        Returns:
            Rendered prompt string
        """
        template = self.env.get_template("clinical_note.jinja2")

        return template.render(
            code=code,
            num_examples=num_examples
        )
```

**Step 5: Run test to verify it passes**

```bash
pytest tests/data/test_prompt_templates.py -v
```

Expected: PASS - All template tests pass

**Step 6: Commit**

```bash
git add .
git commit -m "feat: add prompt templates for synthetic data generation"
```

Expected: Clean commit

---

### Task 7: Synthetic Data Generator

**Files:**
- Create: `src/skstuner/data/synthetic_generator.py`
- Create: `tests/data/test_synthetic_generator.py`
- Create: `scripts/generate_synthetic_data.py`

**Step 1: Write generator test**

Create `tests/data/test_synthetic_generator.py`:
```python
import pytest
from unittest.mock import Mock, patch
from skstuner.data.synthetic_generator import SyntheticDataGenerator
from skstuner.data.sks_parser import SKSCode


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client"""
    with patch('anthropic.Anthropic') as mock:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="""NOTE_1: Patient har jernmangelanæmi
NOTE_2: Kvindelig patient med anæmi pga jerntab
NOTE_3: Diagnosticeret med jernmangel efter blodprøver""")]
        mock_client.messages.create.return_value = mock_response
        mock.return_value = mock_client
        yield mock


def test_generator_generates_examples(mock_anthropic_client):
    """Test that generator creates examples"""
    code = SKSCode(code="D50", description="Jernmangelanæmi", category="D", level=1)

    generator = SyntheticDataGenerator(api_key="test_key")
    examples = generator.generate_for_code(code, num_examples=3)

    assert len(examples) == 3
    assert all('jernmangel' in ex.lower() or 'anæmi' in ex.lower() for ex in examples)


def test_generator_parses_response_format():
    """Test that generator correctly parses NOTE_N format"""
    response_text = """NOTE_1: First note
NOTE_2: Second note
NOTE_3: Third note"""

    generator = SyntheticDataGenerator(api_key="test_key")
    examples = generator._parse_response(response_text)

    assert len(examples) == 3
    assert examples[0] == "First note"
    assert examples[1] == "Second note"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/data/test_synthetic_generator.py -v
```

Expected: FAIL - Module not found

**Step 3: Implement generator**

Create `src/skstuner/data/synthetic_generator.py`:
```python
"""Synthetic clinical note generation using LLMs"""
from typing import List, Optional
import logging
import re
from anthropic import Anthropic
from skstuner.data.sks_parser import SKSCode
from skstuner.data.prompt_templates import PromptTemplateManager

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic clinical notes using Claude"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.template_manager = PromptTemplateManager()

    def generate_for_code(
        self,
        code: SKSCode,
        num_examples: int = 10,
        max_tokens: int = 4000
    ) -> List[str]:
        """
        Generate synthetic clinical notes for a SKS code

        Args:
            code: SKS code to generate examples for
            num_examples: Number of examples to generate
            max_tokens: Maximum tokens in response

        Returns:
            List of generated clinical note texts
        """
        logger.info(f"Generating {num_examples} examples for {code.code}")

        # Render prompt
        prompt = self.template_manager.render_clinical_note_prompt(
            code=code,
            num_examples=num_examples
        )

        # Call Claude API
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response_text = response.content[0].text
            examples = self._parse_response(response_text)

            logger.info(f"Generated {len(examples)} examples for {code.code}")
            return examples

        except Exception as e:
            logger.error(f"Failed to generate examples for {code.code}: {e}")
            return []

    def _parse_response(self, response_text: str) -> List[str]:
        """
        Parse response text to extract clinical notes

        Args:
            response_text: Raw response from LLM

        Returns:
            List of extracted notes
        """
        # Look for NOTE_N: format
        pattern = r'NOTE_\d+:\s*(.+?)(?=NOTE_\d+:|$)'
        matches = re.findall(pattern, response_text, re.DOTALL)

        # Clean up extracted notes
        notes = [match.strip() for match in matches]

        # Filter out empty or very short notes
        notes = [note for note in notes if len(note) > 20]

        return notes

    def generate_dataset(
        self,
        codes: List[SKSCode],
        examples_per_code: int = 10,
        batch_size: int = 10
    ) -> List[dict]:
        """
        Generate complete dataset for list of codes

        Args:
            codes: List of SKS codes
            examples_per_code: Number of examples per code
            batch_size: Number of codes to process before logging

        Returns:
            List of examples with format [{"text": "...", "label": "D50"}, ...]
        """
        dataset = []

        for i, code in enumerate(codes):
            examples = self.generate_for_code(code, num_examples=examples_per_code)

            for example in examples:
                dataset.append({
                    "text": example,
                    "label": code.code,
                    "description": code.description,
                    "category": code.category,
                    "level": code.level
                })

            if (i + 1) % batch_size == 0:
                logger.info(f"Progress: {i + 1}/{len(codes)} codes processed, {len(dataset)} total examples")

        logger.info(f"Dataset generation complete: {len(dataset)} examples from {len(codes)} codes")
        return dataset
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/data/test_synthetic_generator.py -v
```

Expected: PASS - All generator tests pass

**Step 5: Create generation script**

Create `scripts/generate_synthetic_data.py`:
```python
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
```

Make executable:
```bash
chmod +x scripts/generate_synthetic_data.py
```

**Step 6: Commit**

```bash
git add .
git commit -m "feat: add synthetic data generator with CLI"
```

Expected: Clean commit

---

## Phase 4: Model Training Pipeline

### Task 8: Multi-Task Classification Model

**Files:**
- Create: `src/skstuner/models/multi_task_classifier.py`
- Create: `tests/models/test_multi_task_classifier.py`

**Step 1: Write model test**

Create `tests/models/test_multi_task_classifier.py`:
```python
import pytest
import torch
from transformers import AutoTokenizer
from skstuner.models.multi_task_classifier import MultiTaskSKSClassifier


def test_model_initialization():
    """Test that model initializes correctly"""
    model = MultiTaskSKSClassifier(
        model_name="prajjwal1/bert-tiny",  # Tiny model for testing
        num_labels_level1=10,
        num_labels_level2=50,
        num_labels_level3=200,
        num_labels_level4=500
    )

    assert model is not None
    assert hasattr(model, 'classifier_level1')
    assert hasattr(model, 'classifier_level4')


def test_model_forward_pass():
    """Test model forward pass produces correct output shapes"""
    model = MultiTaskSKSClassifier(
        model_name="prajjwal1/bert-tiny",
        num_labels_level1=10,
        num_labels_level2=50,
        num_labels_level3=200,
        num_labels_level4=500
    )

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    inputs = tokenizer("Test text", return_tensors="pt")

    outputs = model(**inputs)

    assert 'logits_level1' in outputs
    assert 'logits_level4' in outputs
    assert outputs['logits_level1'].shape == (1, 10)
    assert outputs['logits_level4'].shape == (1, 500)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/models/test_multi_task_classifier.py -v
```

Expected: FAIL - Module not found

**Step 3: Implement multi-task model**

Create `src/skstuner/models/multi_task_classifier.py`:
```python
"""Multi-task hierarchical classification model"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional


class MultiTaskSKSClassifier(nn.Module):
    """
    Multi-task classifier for hierarchical SKS codes

    Predicts at 4 hierarchy levels simultaneously with shared encoder
    """

    def __init__(
        self,
        model_name: str,
        num_labels_level1: int,
        num_labels_level2: int,
        num_labels_level3: int,
        num_labels_level4: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Load base model
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.config.hidden_size

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Classification heads for each level
        self.classifier_level1 = nn.Linear(hidden_size, num_labels_level1)
        self.classifier_level2 = nn.Linear(hidden_size, num_labels_level2)
        self.classifier_level3 = nn.Linear(hidden_size, num_labels_level3)
        self.classifier_level4 = nn.Linear(hidden_size, num_labels_level4)

        self.num_labels_level1 = num_labels_level1
        self.num_labels_level2 = num_labels_level2
        self.num_labels_level3 = num_labels_level3
        self.num_labels_level4 = num_labels_level4

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels_level1: Optional[torch.Tensor] = None,
        labels_level2: Optional[torch.Tensor] = None,
        labels_level3: Optional[torch.Tensor] = None,
        labels_level4: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels_level1-4: Optional labels for each level

        Returns:
            Dictionary with logits and optional loss for each level
        """
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Get logits from each classification head
        logits_level1 = self.classifier_level1(pooled_output)
        logits_level2 = self.classifier_level2(pooled_output)
        logits_level3 = self.classifier_level3(pooled_output)
        logits_level4 = self.classifier_level4(pooled_output)

        result = {
            'logits_level1': logits_level1,
            'logits_level2': logits_level2,
            'logits_level3': logits_level3,
            'logits_level4': logits_level4,
        }

        # Calculate loss if labels provided
        if labels_level1 is not None:
            loss_fct = nn.CrossEntropyLoss()

            loss_level1 = loss_fct(logits_level1, labels_level1)
            loss_level2 = loss_fct(logits_level2, labels_level2) if labels_level2 is not None else 0
            loss_level3 = loss_fct(logits_level3, labels_level3) if labels_level3 is not None else 0
            loss_level4 = loss_fct(logits_level4, labels_level4) if labels_level4 is not None else 0

            # Weighted sum of losses (can be tuned)
            total_loss = (
                0.1 * loss_level1 +
                0.2 * loss_level2 +
                0.3 * loss_level3 +
                0.4 * loss_level4
            )

            result['loss'] = total_loss
            result['loss_level1'] = loss_level1
            result['loss_level2'] = loss_level2
            result['loss_level3'] = loss_level3
            result['loss_level4'] = loss_level4

        return result
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/models/test_multi_task_classifier.py -v
```

Expected: PASS - All model tests pass

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add multi-task hierarchical classification model"
```

Expected: Clean commit

---

### Task 9: Dataset Preparation

**Files:**
- Create: `src/skstuner/training/dataset.py`
- Create: `tests/training/test_dataset.py`

**Step 1: Write dataset test**

Create `tests/training/test_dataset.py`:
```python
import pytest
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from skstuner.training.dataset import SKSDataset, create_label_mappings


@pytest.fixture
def sample_dataset_file(tmp_path):
    """Create sample dataset file"""
    data = [
        {"text": "Patient har diabetes", "label": "D50", "level": 2},
        {"text": "Akut appendicitis", "label": "K01", "level": 2},
    ]

    file_path = tmp_path / "dataset.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)

    return file_path


def test_create_label_mappings():
    """Test creating label mappings from codes"""
    codes = ["D50", "D500", "D501", "K01"]
    label2id, id2label = create_label_mappings(codes)

    assert len(label2id) == 4
    assert len(id2label) == 4
    assert label2id["D50"] == 0
    assert id2label[0] == "D50"


def test_dataset_loads_data(sample_dataset_file, tmp_path):
    """Test that dataset loads and tokenizes correctly"""
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    dataset = SKSDataset(
        data_path=sample_dataset_file,
        tokenizer=tokenizer,
        max_length=128
    )

    assert len(dataset) == 2

    item = dataset[0]
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert 'labels' in item


def test_dataset_in_dataloader(sample_dataset_file):
    """Test dataset works with DataLoader"""
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")

    dataset = SKSDataset(
        data_path=sample_dataset_file,
        tokenizer=tokenizer,
        max_length=128
    )

    dataloader = DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))

    assert batch['input_ids'].shape[0] == 2
    assert batch['labels'].shape[0] == 2
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/training/test_dataset.py -v
```

Expected: FAIL - Module not found

**Step 3: Implement dataset class**

Create `src/skstuner/training/dataset.py`:
```python
"""Dataset classes for SKS classification"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def create_label_mappings(codes: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create label to ID and ID to label mappings

    Args:
        codes: List of SKS codes

    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    sorted_codes = sorted(set(codes))
    label2id = {code: idx for idx, code in enumerate(sorted_codes)}
    id2label = {idx: code for code, idx in label2id.items()}

    return label2id, id2label


class SKSDataset(Dataset):
    """Dataset for SKS code classification"""

    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        label2id: Dict[str, int] = None
    ):
        """
        Initialize dataset

        Args:
            data_path: Path to JSON file with data
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            label2id: Optional label to ID mapping
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        with open(data_path) as f:
            self.data = json.load(f)

        # Create label mappings if not provided
        if label2id is None:
            all_labels = [item['label'] for item in self.data]
            self.label2id, self.id2label = create_label_mappings(all_labels)
        else:
            self.label2id = label2id
            self.id2label = {v: k for k, v in label2id.items()}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item at index

        Args:
            idx: Index

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        item = self.data[idx]
        text = item['text']
        label = item['label']

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get label ID
        label_id = self.label2id[label]

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }


class HierarchicalSKSDataset(Dataset):
    """Dataset for hierarchical multi-task SKS classification"""

    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizer,
        taxonomy_path: Path,
        max_length: int = 512
    ):
        """
        Initialize hierarchical dataset

        Args:
            data_path: Path to JSON file with data
            tokenizer: Hugging Face tokenizer
            taxonomy_path: Path to taxonomy JSON with label mappings
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        with open(data_path) as f:
            self.data = json.load(f)

        # Load taxonomy
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)

        self.label2id = taxonomy['label2id']
        self.id2label = taxonomy['id2label']

    def _get_hierarchy_labels(self, code: str) -> Dict[str, int]:
        """
        Extract labels for all hierarchy levels from full code

        Args:
            code: Full SKS code (e.g., "D500")

        Returns:
            Dictionary with label IDs for each level
        """
        # Extract codes at each level
        # Level 1: First character or two (D or D5)
        # Level 2: First 3 chars (D50)
        # Level 3: First 4 chars (D500)
        # Level 4: Full code

        code_clean = code.strip()

        labels = {}

        # Determine which levels exist for this code
        if len(code_clean) >= 1:
            level1_code = code_clean[0]
            labels['level1'] = self.label2id.get(level1_code, 0)

        if len(code_clean) >= 2:
            level2_code = code_clean[:2]
            labels['level2'] = self.label2id.get(level2_code, 0)

        if len(code_clean) >= 3:
            level3_code = code_clean[:3]
            labels['level3'] = self.label2id.get(level3_code, 0)

        if len(code_clean) >= 3:
            labels['level4'] = self.label2id.get(code_clean, 0)

        return labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item with hierarchical labels

        Args:
            idx: Index

        Returns:
            Dictionary with input_ids, attention_mask, and labels for each level
        """
        item = self.data[idx]
        text = item['text']
        label = item['label']

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get hierarchical labels
        hierarchy_labels = self._get_hierarchy_labels(label)

        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

        # Add labels for each level
        for level, label_id in hierarchy_labels.items():
            result[f'labels_{level}'] = torch.tensor(label_id, dtype=torch.long)

        return result
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/training/test_dataset.py -v
```

Expected: PASS - All dataset tests pass

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add dataset classes for training"
```

Expected: Clean commit

---

**Due to length constraints, I'll continue with remaining phases in condensed format. The pattern remains: Test → Fail → Implement → Pass → Commit for each component.**

---

### Task 10: Training Loop

**Files:**
- Create: `src/skstuner/training/trainer.py`
- Create: `scripts/train.py`

**Implementation includes:**
- Model-agnostic training loop
- Support for encoder and decoder models
- LoRA integration for large models
- Weights & Biases logging
- Checkpoint saving
- Early stopping

**Commit:** `feat: add training loop with W&B integration`

---

### Task 11: Evaluation Metrics

**Files:**
- Create: `src/skstuner/evaluation/metrics.py`
- Create: `tests/evaluation/test_metrics.py`

**Implementation includes:**
- Accuracy (exact match, top-k)
- Per-level accuracy
- Hierarchical F1
- Per-class metrics
- Confusion matrix generation

**Commit:** `feat: add comprehensive evaluation metrics`

---

### Task 12: Inference API

**Files:**
- Create: `src/skstuner/api/main.py`
- Create: `src/skstuner/api/models.py` (Pydantic schemas)
- Create: `tests/api/test_api.py`

**Implementation includes:**
- FastAPI service with `/classify` endpoint
- Model loading and caching
- Batch prediction support
- Model switching via config
- Health check endpoint

**Commit:** `feat: add FastAPI inference service`

---

## Execution Instructions

After all tasks are complete, the system can be used as follows:

```bash
# 1. Download SKS codes
python scripts/download_sks.py

# 2. Process codes
python scripts/process_sks.py

# 3. Generate synthetic data (start with subset for testing)
python scripts/generate_synthetic_data.py --max-codes 100 --examples-per-code 10

# 4. Train model
python scripts/train.py --model xlm-roberta-large --epochs 5

# 5. Run inference API
uvicorn src.skstuner.api.main:app --host 0.0.0.0 --port 8000
```

## Testing Strategy

Run tests frequently:
```bash
# All tests
pytest -v

# With coverage
pytest --cov=src/skstuner --cov-report=html

# Specific module
pytest tests/data/ -v
```

## Next Steps After PoC

1. Collect real labeled data for validation
2. Iteratively improve synthetic data quality
3. Experiment with different model architectures
4. Fine-tune hyperparameters
5. Add more robust error handling
6. Deploy to production environment

---

**Plan saved to:** `docs/plans/2025-11-07-sks-classification-system.md`
