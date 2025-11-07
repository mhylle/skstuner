"""Synthetic clinical note generation using LLMs"""
from typing import List
import logging
import re
from anthropic import Anthropic
from skstuner.data.sks_parser import SKSCode
from skstuner.data.prompt_templates import PromptTemplateManager

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic clinical notes using Claude"""

    MIN_NOTE_LENGTH = 20  # Minimum characters to filter out incomplete/empty notes

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.template_manager = PromptTemplateManager()

    def generate_for_code(
        self,
        code: SKSCode,
        num_examples: int = 10,
        max_tokens: int = 4000
    ) -> List[str]:
        """
        Generate synthetic clinical notes for a SKS code

        Args:
            code: SKS code to generate examples for
            num_examples: Number of examples to generate
            max_tokens: Maximum tokens in response

        Returns:
            List of generated clinical note texts
        """
        logger.info(f"Generating {num_examples} examples for {code.code}")

        # Render prompt
        prompt = self.template_manager.render_clinical_note_prompt(
            code=code,
            num_examples=num_examples
        )

        # Call Claude API
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            if not response.content or not hasattr(response.content[0], 'text'):
                logger.error(f"Unexpected API response format for {code.code}")
                return []

            response_text = response.content[0].text
            examples = self._parse_response(response_text)

            logger.info(f"Generated {len(examples)} examples for {code.code}")
            return examples

        except Exception as e:
            logger.error(f"Failed to generate examples for {code.code}: {e}")
            return []

    def _parse_response(self, response_text: str) -> List[str]:
        """
        Parse response text to extract clinical notes

        Args:
            response_text: Raw response from LLM

        Returns:
            List of extracted notes
        """
        # Look for NOTE_N: format
        pattern = r'NOTE_\d+:\s*(.+?)(?=NOTE_\d+:|$)'
        matches = re.findall(pattern, response_text, re.DOTALL)

        # Clean up extracted notes
        notes = [match.strip() for match in matches]

        # Filter out empty or very short notes
        notes = [note for note in notes if len(note) > self.MIN_NOTE_LENGTH]

        return notes

    def generate_dataset(
        self,
        codes: List[SKSCode],
        examples_per_code: int = 10,
        batch_size: int = 10
    ) -> List[dict]:
        """
        Generate complete dataset for list of codes

        Args:
            codes: List of SKS codes
            examples_per_code: Number of examples per code
            batch_size: Number of codes to process before logging

        Returns:
            List of examples with format [{"text": "...", "label": "D50"}, ...]
        """
        dataset = []

        for i, code in enumerate(codes):
            examples = self.generate_for_code(code, num_examples=examples_per_code)

            for example in examples:
                dataset.append({
                    "text": example,
                    "label": code.code,
                    "description": code.description,
                    "category": code.category,
                    "level": code.level
                })

            if (i + 1) % batch_size == 0:
                logger.info(f"Progress: {i + 1}/{len(codes)} codes processed, {len(dataset)} total examples")

        logger.info(f"Dataset generation complete: {len(dataset)} examples from {len(codes)} codes")
        return dataset
