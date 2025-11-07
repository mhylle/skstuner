import pytest
from unittest.mock import Mock, patch
from skstuner.data.synthetic_generator import SyntheticDataGenerator
from skstuner.data.sks_parser import SKSCode


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client"""
    with patch('skstuner.data.synthetic_generator.Anthropic') as mock:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="""NOTE_1: Patient har jernmangelanæmi
NOTE_2: Kvindelig patient med anæmi pga jerntab
NOTE_3: Diagnosticeret med jernmangel efter blodprøver""")]
        mock_client.messages.create.return_value = mock_response
        mock.return_value = mock_client
        yield mock


def test_generator_generates_examples(mock_anthropic_client):
    """Test that generator creates examples"""
    code = SKSCode(code="D50", description="Jernmangelanæmi", category="D", level=1)

    generator = SyntheticDataGenerator(api_key="test_key")
    examples = generator.generate_for_code(code, num_examples=3)

    assert len(examples) == 3
    assert all('jernmangel' in ex.lower() or 'anæmi' in ex.lower() for ex in examples)


def test_generator_parses_response_format():
    """Test that generator correctly parses NOTE_N format"""
    response_text = """NOTE_1: First note with enough characters to pass filter
NOTE_2: Second note with enough characters to pass filter
NOTE_3: Third note with enough characters to pass filter"""

    generator = SyntheticDataGenerator(api_key="test_key")
    examples = generator._parse_response(response_text)

    assert len(examples) == 3
    assert examples[0] == "First note with enough characters to pass filter"
    assert examples[1] == "Second note with enough characters to pass filter"
