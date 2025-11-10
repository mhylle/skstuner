from skstuner.data.prompt_templates import PromptTemplateManager
from skstuner.data.sks_parser import SKSCode


def test_template_manager_renders_prompt():
    """Test that template manager renders prompts correctly"""
    manager = PromptTemplateManager()

    code = SKSCode(code="D50", description="Jernmangelanæmi", category="D", level=1)

    prompt = manager.render_clinical_note_prompt(code, num_examples=5)

    assert "D50" in prompt
    assert "Jernmangelanæmi" in prompt
    assert "5" in prompt or "fem" in prompt.lower()


def test_template_variations_specified():
    """Test that template includes variation instructions"""
    manager = PromptTemplateManager()

    code = SKSCode(code="D50", description="Jernmangelanæmi", category="D", level=1)
    prompt = manager.render_clinical_note_prompt(code, num_examples=3)

    # Should include instructions for variations
    assert any(word in prompt.lower() for word in ["variation", "different", "diverse"])
