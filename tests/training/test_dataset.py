"""Tests for dataset preparation."""

import json
import tempfile
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from skstuner.training.dataset import SKSDataset, prepare_datasets


@pytest.fixture
def sample_data():
    """Create sample training data."""
    return [
        {
            "text": "Patient har diabetes type 2 og hypertension",
            "sks_code": "DE11",
            "description": "Type 2-diabetes",
            "category": "D",
        },
        {
            "text": "Patient diagnosticeret med asthma bronchiale",
            "sks_code": "DJ45",
            "description": "Astma",
            "category": "D",
        },
        {
            "text": "Akut myokardieinfarkt, anteriort",
            "sks_code": "DI21",
            "description": "Akut myokardieinfarkt",
            "category": "D",
        },
    ]


@pytest.fixture
def data_file(sample_data):
    """Create temporary data file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_data, f)
        return Path(f.name)


@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing."""
    # Use a small, fast model for testing
    return AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")


def test_sks_dataset_initialization(data_file, tokenizer):
    """Test SKSDataset initialization."""
    dataset = SKSDataset(
        data_path=data_file,
        tokenizer=tokenizer,
        max_length=128,
    )

    assert len(dataset.examples) == 3
    assert dataset.num_labels == 3
    assert "DE11" in dataset.label_to_id
    assert "DJ45" in dataset.label_to_id
    assert "DI21" in dataset.label_to_id


def test_sks_dataset_label_mapping(data_file, tokenizer):
    """Test label mapping creation."""
    dataset = SKSDataset(
        data_path=data_file,
        tokenizer=tokenizer,
        max_length=128,
    )

    # Check bidirectional mapping
    for code, idx in dataset.label_to_id.items():
        assert dataset.id_to_label[idx] == code


def test_sks_dataset_to_dataframe(data_file, tokenizer):
    """Test conversion to DataFrame."""
    dataset = SKSDataset(
        data_path=data_file,
        tokenizer=tokenizer,
        max_length=128,
    )

    df = dataset.to_dataframe()

    assert len(df) == 3
    assert "text" in df.columns
    assert "sks_code" in df.columns
    assert "label_id" in df.columns
    assert all(df["label_id"] >= 0)


def test_sks_dataset_to_hf_dataset(data_file, tokenizer):
    """Test conversion to HuggingFace Dataset."""
    dataset = SKSDataset(
        data_path=data_file,
        tokenizer=tokenizer,
        max_length=128,
    )

    hf_dataset = dataset.to_hf_dataset()

    assert len(hf_dataset) == 3
    assert "input_ids" in hf_dataset.column_names
    assert "attention_mask" in hf_dataset.column_names
    assert "labels" in hf_dataset.column_names


def test_prepare_datasets(data_file, tokenizer):
    """Test dataset preparation with train/val/test split."""
    datasets, label_to_id, id_to_label = prepare_datasets(
        data_path=data_file,
        tokenizer=tokenizer,
        max_length=128,
        test_size=0.33,
        val_size=0.5,
        random_state=42,
    )

    # Check that we have all splits
    assert "train" in datasets
    assert "validation" in datasets
    assert "test" in datasets

    # Check total samples
    total = len(datasets["train"]) + len(datasets["validation"]) + len(datasets["test"])
    assert total == 3

    # Check label mappings
    assert len(label_to_id) == 3
    assert len(id_to_label) == 3


def test_sks_dataset_with_dict_format(tokenizer):
    """Test loading data in dict format with 'examples' key."""
    data = {"examples": [{"text": "Test", "sks_code": "TEST"}]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        data_file = Path(f.name)

    dataset = SKSDataset(
        data_path=data_file,
        tokenizer=tokenizer,
        max_length=128,
    )

    assert len(dataset.examples) == 1
    data_file.unlink()
