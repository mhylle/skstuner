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

    # LLM Provider settings
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "claude"))
    claude_model: str = field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "https://ollama:11434")
    )
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.3"))
    ollama_timeout: int = field(
        default_factory=lambda: int(os.getenv("OLLAMA_TIMEOUT", "120"))
    )

    # Project settings
    wandb_project: str = field(default_factory=lambda: os.getenv("WANDB_PROJECT", "skstuner"))
    wandb_entity: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_ENTITY"))

    # Paths
    root_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Initialize computed fields and create necessary directories"""
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

    def validate_api_key(self, key_name: str) -> None:
        """
        Validate that a required API key is set

        Args:
            key_name: Name of the API key field to validate

        Raises:
            ValueError: If the API key is not set or is empty
        """
        key_value = getattr(self, key_name, "")
        if not key_value or key_value.strip() == "":
            raise ValueError(
                f"{key_name.upper()} is not set. "
                f"Please set it in your .env file or environment variables."
            )


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
        """
        Load model config from YAML file

        Args:
            path: Path to YAML configuration file

        Returns:
            ModelConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is malformed or missing required fields
        """
        if not path.exists():
            raise FileNotFoundError(f"Model config file not found: {path}")

        try:
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {path}: {e}")

        if not config_dict:
            raise ValueError(f"Empty config file: {path}")

        # Convert learning_rate to float if it's a string
        if "learning_rate" in config_dict and isinstance(config_dict["learning_rate"], str):
            config_dict["learning_rate"] = float(config_dict["learning_rate"])

        return cls(**config_dict)

    def to_yaml(self, path: Path) -> None:
        """
        Save model config to YAML file

        Args:
            path: Path where to save the configuration
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
