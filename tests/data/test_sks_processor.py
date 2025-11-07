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
