import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from skstuner.data.sks_downloader import SKSDownloader


def test_sks_downloader_downloads_file(tmp_path):
    """Test that SKS downloader downloads the file"""
    downloader = SKSDownloader(output_dir=tmp_path)

    # Mock the HTTP request
    with patch('skstuner.data.sks_downloader.requests.get') as mock_get:
        mock_response = Mock()
        # Create test data with multiple lines (at least 10) and long enough lines
        test_content = "\n".join([
            "Test SKS data line 1 with enough content to pass validation checks.",
            "Test SKS data line 2 with enough content to pass validation checks.",
            "Test SKS data line 3 with enough content to pass validation checks.",
            "Test SKS data line 4 with enough content to pass validation checks.",
            "Test SKS data line 5 with enough content to pass validation checks.",
            "Test SKS data line 6 with enough content to pass validation checks.",
            "Test SKS data line 7 with enough content to pass validation checks.",
            "Test SKS data line 8 with enough content to pass validation checks.",
            "Test SKS data line 9 with enough content to pass validation checks.",
            "Test SKS data line 10 with enough content to pass validation checks.",
        ])
        mock_response.content = test_content.encode('latin-1')
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        output_file = downloader.download()

        assert output_file.exists()
        assert output_file.read_bytes() == test_content.encode('latin-1')


def test_sks_downloader_validates_file_format(tmp_path):
    """Test that downloader validates SKS file format"""
    test_file = tmp_path / "SKScomplete.txt"
    test_file.write_text("Invalid format")

    downloader = SKSDownloader(output_dir=tmp_path)

    with pytest.raises(ValueError, match="Invalid SKS file format"):
        downloader.validate_file(test_file)
