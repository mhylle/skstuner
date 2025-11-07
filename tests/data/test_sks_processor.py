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


def test_processor_exports_taxonomy(tmp_path, sample_codes):
    """Test that processor exports taxonomy for model training"""
    processor = SKSProcessor(codes=sample_codes)
    output_file = tmp_path / "taxonomy.json"

    processor.export_taxonomy(output_file)

    assert output_file.exists()

    with open(output_file) as f:
        taxonomy = json.load(f)

    # Verify structure
    assert taxonomy['num_labels'] == 3
    assert len(taxonomy['label2id']) == 3
    assert len(taxonomy['id2label']) == 3
    assert len(taxonomy['descriptions']) == 3

    # Verify categories and levels exist
    assert 'D' in taxonomy['categories']
    assert 'K' in taxonomy['categories']


def test_processor_filters_by_level(sample_codes):
    """Test filtering codes by hierarchy level"""
    processor = SKSProcessor(codes=sample_codes)
    level_1_codes = processor.filter_by_level(1)
    level_2_codes = processor.filter_by_level(2)

    assert len(level_1_codes) == 2
    assert all(code.level == 1 for code in level_1_codes)

    assert len(level_2_codes) == 1
    assert all(code.level == 2 for code in level_2_codes)
