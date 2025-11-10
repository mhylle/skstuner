import pytest
from skstuner.data.sks_parser import SKSParser, SKSCode


@pytest.fixture
def sample_sks_file(tmp_path):
    """Create a sample SKS file for testing"""
    sample_file = tmp_path / "sample_sks.txt"
    sample_content = """D50  Jernmangelanæmi                                                  2018-01-01          9999-12-312018-01-01          9999-12-31D     001
D500 Jernmangelanæmi efter blødning                                   2018-01-01          9999-12-312018-01-01          9999-12-31D     002                          """
    sample_file.write_text(sample_content, encoding="latin-1")
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
        SKSCode(code="D50", description="Jernmangelanæmi", category="D", level=1, parent_code=None),
        SKSCode(
            code="D500",
            description="Jernmangelanæmi efter blødning",
            category="D",
            level=2,
            parent_code="D50",
        ),
        SKSCode(
            code="D501",
            description="Sideropenisk dysfagi",
            category="D",
            level=2,
            parent_code="D50",
        ),
    ]

    parser = SKSParser()
    hierarchy = parser.build_hierarchy(codes)

    assert "D50" in hierarchy
    assert len(hierarchy["D50"]["children"]) == 2
    assert "D500" in hierarchy["D50"]["children"]


def test_determine_levels_correctly():
    """Test level determination for various code structures"""
    parser = SKSParser()

    # Test different code lengths map to correct levels
    assert parser._determine_level("D50") == 1
    assert parser._determine_level("D500") == 2
    assert parser._determine_level("D50A") == 2
    assert parser._determine_level("D500A") == 3
