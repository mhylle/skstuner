"""Tests for model creation and configuration."""

import tempfile
from pathlib import Path

import pytest
import yaml

from skstuner.training.model import (
    load_model_config,
    _get_lora_target_modules,
)


@pytest.fixture
def sample_config():
    """Create sample model configuration."""
    return {
        "model_name": "prajjwal1/bert-tiny",
        "model_type": "encoder",
        "num_labels": 10,
        "hidden_size": 128,
        "learning_rate": 2e-5,
        "batch_size": 8,
        "num_epochs": 3,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "max_length": 128,
        "gradient_accumulation_steps": 1,
        "use_lora": False,
    }


@pytest.fixture
def config_file(sample_config):
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config, f)
        return Path(f.name)


def test_load_model_config(config_file, sample_config):
    """Test loading model configuration from YAML."""
    config = load_model_config(config_file)

    assert config["model_name"] == sample_config["model_name"]
    assert config["model_type"] == sample_config["model_type"]
    assert config["num_labels"] == sample_config["num_labels"]
    assert config["learning_rate"] == sample_config["learning_rate"]

    config_file.unlink()


def test_get_lora_target_modules_xlm_roberta():
    """Test LoRA target modules for XLM-RoBERTa."""
    modules = _get_lora_target_modules("xlm-roberta-base", "encoder")
    assert "query" in modules
    assert "value" in modules


def test_get_lora_target_modules_phi():
    """Test LoRA target modules for Phi models."""
    modules = _get_lora_target_modules("microsoft/phi-3-mini", "decoder")
    assert "q_proj" in modules
    assert "k_proj" in modules
    assert "v_proj" in modules
    assert "o_proj" in modules


def test_get_lora_target_modules_gemma():
    """Test LoRA target modules for Gemma models."""
    modules = _get_lora_target_modules("google/gemma-7b", "decoder")
    assert "q_proj" in modules
    assert "k_proj" in modules
    assert "v_proj" in modules
    assert "o_proj" in modules


def test_get_lora_target_modules_llama():
    """Test LoRA target modules for LLaMA models."""
    modules = _get_lora_target_modules("meta-llama/Llama-2-7b", "decoder")
    assert "q_proj" in modules
    assert "k_proj" in modules
    assert "v_proj" in modules
    assert "o_proj" in modules


def test_get_lora_target_modules_default_decoder():
    """Test LoRA target modules for unknown decoder model."""
    modules = _get_lora_target_modules("unknown/model", "decoder")
    assert "q_proj" in modules or "v_proj" in modules


def test_get_lora_target_modules_default_encoder():
    """Test LoRA target modules for unknown encoder model."""
    modules = _get_lora_target_modules("unknown/model", "encoder")
    assert "query" in modules or "value" in modules
