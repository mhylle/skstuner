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
