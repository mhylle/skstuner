import pytest
from unittest.mock import Mock, patch
from skstuner.data.synthetic_generator import SyntheticDataGenerator
from skstuner.data.sks_parser import SKSCode
from skstuner.data.llm_providers import LLMProvider


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""

    def __init__(self, response_text: str = None):
        self.response_text = response_text or """NOTE_1: Patient har jernmangelanæmi
NOTE_2: Kvindelig patient med anæmi pga jerntab
NOTE_3: Diagnosticeret med jernmangel efter blodprøver"""

    def generate(self, prompt: str, max_tokens: int = 4000) -> str:
        return self.response_text

    def get_model_name(self) -> str:
        return "mock-model"


@pytest.fixture
def mock_provider():
    """Mock LLM provider"""
    return MockLLMProvider()


def test_generator_generates_examples(mock_provider):
    """Test that generator creates examples"""
    code = SKSCode(code="D50", description="Jernmangelanæmi", category="D", level=1)

    generator = SyntheticDataGenerator(provider=mock_provider)
    examples = generator.generate_for_code(code, num_examples=3)

    assert len(examples) == 3
    assert all("jernmangel" in ex.lower() or "anæmi" in ex.lower() for ex in examples)


def test_generator_parses_response_format():
    """Test that generator correctly parses NOTE_N format"""
    response_text = """NOTE_1: First note with enough characters to pass filter
NOTE_2: Second note with enough characters to pass filter
NOTE_3: Third note with enough characters to pass filter"""

    mock_provider = MockLLMProvider(response_text=response_text)
    generator = SyntheticDataGenerator(provider=mock_provider)
    examples = generator._parse_response(response_text)

    assert len(examples) == 3
    assert examples[0] == "First note with enough characters to pass filter"
    assert examples[1] == "Second note with enough characters to pass filter"
