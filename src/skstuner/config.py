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

        # Convert learning_rate to float if it's a string
        if "learning_rate" in config_dict and isinstance(config_dict["learning_rate"], str):
            config_dict["learning_rate"] = float(config_dict["learning_rate"])

        return cls(**config_dict)

    def to_yaml(self, path: Path):
        """Save model config to YAML file"""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
