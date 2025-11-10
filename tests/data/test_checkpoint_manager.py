"""Tests for checkpoint manager"""

import pytest
from pathlib import Path
from skstuner.data.checkpoint_manager import CheckpointManager


@pytest.fixture
def temp_checkpoint_file(tmp_path):
    """Fixture for temporary checkpoint file"""
    return tmp_path / "test_checkpoint.json"


@pytest.fixture
def checkpoint_manager(temp_checkpoint_file):
    """Fixture for checkpoint manager"""
    return CheckpointManager(temp_checkpoint_file)


def test_checkpoint_manager_init(checkpoint_manager, temp_checkpoint_file):
    """Test checkpoint manager initialization"""
    assert checkpoint_manager.checkpoint_file == temp_checkpoint_file
    assert len(checkpoint_manager.processed_codes) == 0
    assert len(checkpoint_manager.failed_codes) == 0
    assert len(checkpoint_manager.dataset) == 0


def test_save_and_load_checkpoint(checkpoint_manager):
    """Test saving and loading checkpoint"""
    # Add some data
    checkpoint_manager.mark_code_processed("D50")
    checkpoint_manager.mark_code_processed("D51")
    checkpoint_manager.mark_code_failed("D52")
    checkpoint_manager.add_examples([{"text": "example1", "label": "D50"}])

    # Save checkpoint
    checkpoint_manager.save()
    assert checkpoint_manager.checkpoint_file.exists()

    # Create new manager and load
    new_manager = CheckpointManager(checkpoint_manager.checkpoint_file)
    loaded = new_manager.load()

    assert loaded is True
    assert "D50" in new_manager.processed_codes
    assert "D51" in new_manager.processed_codes
    assert "D52" in new_manager.failed_codes
    assert len(new_manager.dataset) == 1
    assert new_manager.dataset[0]["text"] == "example1"


def test_load_nonexistent_checkpoint(checkpoint_manager):
    """Test loading when checkpoint doesn't exist"""
    loaded = checkpoint_manager.load()
    assert loaded is False


def test_is_code_processed(checkpoint_manager):
    """Test checking if code is processed"""
    assert checkpoint_manager.is_code_processed("D50") is False

    checkpoint_manager.mark_code_processed("D50")
    assert checkpoint_manager.is_code_processed("D50") is True


def test_get_remaining_codes(checkpoint_manager):
    """Test getting remaining codes"""
    all_codes = ["D50", "D51", "D52", "D53"]

    # Initially all codes are remaining
    remaining = checkpoint_manager.get_remaining_codes(all_codes)
    assert len(remaining) == 4

    # Mark some as processed
    checkpoint_manager.mark_code_processed("D50")
    checkpoint_manager.mark_code_processed("D52")

    remaining = checkpoint_manager.get_remaining_codes(all_codes)
    assert len(remaining) == 2
    assert "D51" in remaining
    assert "D53" in remaining


def test_add_examples(checkpoint_manager):
    """Test adding examples"""
    examples = [
        {"text": "example1", "label": "D50"},
        {"text": "example2", "label": "D50"},
    ]

    checkpoint_manager.add_examples(examples)
    assert len(checkpoint_manager.dataset) == 2


def test_get_statistics(checkpoint_manager):
    """Test getting statistics"""
    checkpoint_manager.mark_code_processed("D50")
    checkpoint_manager.mark_code_processed("D51")
    checkpoint_manager.mark_code_failed("D52")
    checkpoint_manager.add_examples([{"text": "example1", "label": "D50"}])

    stats = checkpoint_manager.get_statistics()
    assert stats["processed_codes"] == 2
    assert stats["failed_codes"] == 1
    assert stats["total_examples"] == 1


def test_delete_checkpoint(checkpoint_manager):
    """Test deleting checkpoint"""
    checkpoint_manager.save()
    assert checkpoint_manager.checkpoint_file.exists()

    checkpoint_manager.delete()
    assert not checkpoint_manager.checkpoint_file.exists()


def test_delete_nonexistent_checkpoint(checkpoint_manager):
    """Test deleting when checkpoint doesn't exist"""
    # Should not raise error
    checkpoint_manager.delete()
    assert not checkpoint_manager.checkpoint_file.exists()


def test_atomic_save(checkpoint_manager):
    """Test that save is atomic"""
    # Add some data and save
    checkpoint_manager.mark_code_processed("D50")
    checkpoint_manager.save()

    # Verify file exists and is valid JSON
    assert checkpoint_manager.checkpoint_file.exists()
    with open(checkpoint_manager.checkpoint_file, "r") as f:
        import json

        data = json.load(f)
        assert "D50" in data["processed_codes"]


def test_checkpoint_metadata(checkpoint_manager):
    """Test that metadata is saved"""
    checkpoint_manager.metadata["test_key"] = "test_value"
    checkpoint_manager.save()

    new_manager = CheckpointManager(checkpoint_manager.checkpoint_file)
    new_manager.load()

    assert "test_key" in new_manager.metadata
    assert new_manager.metadata["test_key"] == "test_value"
    assert "last_updated" in new_manager.metadata
