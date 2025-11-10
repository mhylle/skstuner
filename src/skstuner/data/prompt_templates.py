"""Prompt templates for synthetic data generation"""

from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from skstuner.data.sks_parser import SKSCode


class PromptTemplateManager:
    """Manages Jinja2 templates for LLM prompts"""

    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        if not template_dir.exists():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))

    def render_clinical_note_prompt(self, code: SKSCode, num_examples: int = 10) -> str:
        """
        Render prompt for generating clinical notes

        Args:
            code: SKS code to generate notes for
            num_examples: Number of examples to generate

        Returns:
            Rendered prompt string
        """
        template = self.env.get_template("clinical_note.jinja2")

        return template.render(code=code, num_examples=num_examples)
