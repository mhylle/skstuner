"""Synthetic clinical note generation using LLMs"""

from typing import List, Dict
import logging
import re
from anthropic import Anthropic, APIError
from skstuner.data.sks_parser import SKSCode
from skstuner.data.prompt_templates import PromptTemplateManager

logger = logging.getLogger(__name__)

# Constants
MIN_NOTE_LENGTH = 20  # Minimum characters to filter out incomplete/empty notes
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 4000


class SyntheticDataGenerationError(Exception):
    """Custom exception for synthetic data generation errors"""

    pass


class SyntheticDataGenerator:
    """Generate synthetic clinical notes using Claude"""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        """
        Initialize the synthetic data generator

        Args:
            api_key: Anthropic API key
            model: Claude model to use for generation

        Raises:
            ValueError: If API key is empty or invalid
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")

        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.template_manager = PromptTemplateManager()

    def generate_for_code(
        self, code: SKSCode, num_examples: int = 10, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> List[str]:
        """
        Generate synthetic clinical notes for a SKS code

        Args:
            code: SKS code to generate examples for
            num_examples: Number of examples to generate
            max_tokens: Maximum tokens in response

        Returns:
            List of generated clinical note texts

        Raises:
            SyntheticDataGenerationError: If generation fails
            ValueError: If parameters are invalid
        """
        if num_examples <= 0:
            raise ValueError(f"num_examples must be positive, got {num_examples}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        logger.info(f"Generating {num_examples} examples for {code.code}")

        # Render prompt
        try:
            prompt = self.template_manager.render_clinical_note_prompt(
                code=code, num_examples=num_examples
            )
        except Exception as e:
            raise SyntheticDataGenerationError(
                f"Failed to render prompt template for {code.code}: {e}"
            ) from e

        # Call Claude API
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            if not response.content or not hasattr(response.content[0], "text"):
                raise SyntheticDataGenerationError(
                    f"Unexpected API response format for {code.code}. " f"Response: {response}"
                )

            response_text = response.content[0].text
            examples = self._parse_response(response_text)

            logger.info(f"Generated {len(examples)} valid examples for {code.code}")
            return examples

        except APIError as e:
            raise SyntheticDataGenerationError(
                f"Anthropic API error while generating examples for {code.code}: {e}"
            ) from e
        except Exception as e:
            raise SyntheticDataGenerationError(
                f"Unexpected error generating examples for {code.code}: {e}"
            ) from e

    def _parse_response(self, response_text: str) -> List[str]:
        """
        Parse response text to extract clinical notes

        Args:
            response_text: Raw response from LLM

        Returns:
            List of extracted notes
        """
        # Look for NOTE_N: format
        pattern = r"NOTE_\d+:\s*(.+?)(?=NOTE_\d+:|$)"
        matches = re.findall(pattern, response_text, re.DOTALL)

        # Clean up extracted notes
        notes = [match.strip() for match in matches]

        # Filter out empty or very short notes
        notes = [note for note in notes if len(note) > MIN_NOTE_LENGTH]

        return notes

    def generate_dataset(
        self,
        codes: List[SKSCode],
        examples_per_code: int = 10,
        batch_size: int = 10,
        continue_on_error: bool = True,
    ) -> List[Dict]:
        """
        Generate complete dataset for list of codes

        Args:
            codes: List of SKS codes
            examples_per_code: Number of examples per code
            batch_size: Number of codes to process before logging
            continue_on_error: If True, continue processing on errors; if False, raise

        Returns:
            List of examples with format [{"text": "...", "label": "D50"}, ...]

        Raises:
            SyntheticDataGenerationError: If generation fails and continue_on_error is False
            ValueError: If parameters are invalid
        """
        if not codes:
            raise ValueError("codes list cannot be empty")
        if examples_per_code <= 0:
            raise ValueError(f"examples_per_code must be positive, got {examples_per_code}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        dataset = []
        failed_codes = []

        for i, code in enumerate(codes):
            try:
                examples = self.generate_for_code(code, num_examples=examples_per_code)

                for example in examples:
                    dataset.append(
                        {
                            "text": example,
                            "label": code.code,
                            "description": code.description,
                            "category": code.category,
                            "level": code.level,
                        }
                    )

            except SyntheticDataGenerationError as e:
                logger.error(f"Failed to generate examples for {code.code}: {e}")
                failed_codes.append(code.code)
                if not continue_on_error:
                    raise

            if (i + 1) % batch_size == 0:
                logger.info(
                    f"Progress: {i + 1}/{len(codes)} codes processed, "
                    f"{len(dataset)} total examples, {len(failed_codes)} failures"
                )

        logger.info(
            f"Dataset generation complete: {len(dataset)} examples from {len(codes)} codes. "
            f"Failed codes: {len(failed_codes)}"
        )

        if failed_codes:
            logger.warning(
                f"Failed to generate data for codes: {', '.join(failed_codes[:10])}"
                f"{'...' if len(failed_codes) > 10 else ''}"
            )

        return dataset
