"""Synthetic clinical note generation using LLMs"""

from typing import List, Dict, Optional
import logging
import re
from skstuner.data.sks_parser import SKSCode
from skstuner.data.prompt_templates import PromptTemplateManager
from skstuner.data.checkpoint_manager import CheckpointManager
from skstuner.data.quality_validator import QualityValidator
from skstuner.data.llm_providers import LLMProvider

logger = logging.getLogger(__name__)

# Constants
MIN_NOTE_LENGTH = 20  # Minimum characters to filter out incomplete/empty notes
DEFAULT_MAX_TOKENS = 4000


class SyntheticDataGenerationError(Exception):
    """Custom exception for synthetic data generation errors"""

    pass


class SyntheticDataGenerator:
    """Generate synthetic clinical notes using LLMs"""

    def __init__(self, provider: LLMProvider):
        """
        Initialize the synthetic data generator

        Args:
            provider: LLM provider instance (ClaudeProvider or OllamaProvider)

        Raises:
            ValueError: If provider is None
        """
        if provider is None:
            raise ValueError("Provider cannot be None")

        self.provider = provider
        self.template_manager = PromptTemplateManager()
        logger.info(f"Initialized SyntheticDataGenerator with provider: {provider.get_model_name()}")

    def generate_for_code(
        self,
        code: SKSCode,
        num_examples: int = 10,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        quality_validator: Optional[QualityValidator] = None,
    ) -> List[str]:
        """
        Generate synthetic clinical notes for a SKS code

        Args:
            code: SKS code to generate examples for
            num_examples: Number of examples to generate
            max_tokens: Maximum tokens in response
            quality_validator: Optional quality validator to filter examples

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

        # Call LLM provider
        try:
            response_text = self.provider.generate(prompt=prompt, max_tokens=max_tokens)
            examples = self._parse_response(response_text)

            # Apply quality validation if provided
            if quality_validator:
                validated_examples = []
                rejected_count = 0
                for example in examples:
                    quality = quality_validator.validate(example, code.description)
                    if quality.passed:
                        validated_examples.append(example)
                    else:
                        rejected_count += 1
                        logger.debug(
                            f"Rejected example for {code.code}: {quality.issues}"
                        )
                examples = validated_examples
                if rejected_count > 0:
                    logger.info(
                        f"Quality filter: kept {len(examples)}/{len(examples) + rejected_count} "
                        f"examples for {code.code}"
                    )

            logger.info(f"Generated {len(examples)} valid examples for {code.code}")
            return examples

        except Exception as e:
            raise SyntheticDataGenerationError(
                f"LLM provider error while generating examples for {code.code}: {e}"
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
        checkpoint_manager: Optional[CheckpointManager] = None,
        checkpoint_interval: int = 5,
        quality_validator: Optional[QualityValidator] = None,
    ) -> List[Dict]:
        """
        Generate complete dataset for list of codes

        Args:
            codes: List of SKS codes
            examples_per_code: Number of examples per code
            batch_size: Number of codes to process before logging
            continue_on_error: If True, continue processing on errors; if False, raise
            checkpoint_manager: Optional checkpoint manager for resume capability
            checkpoint_interval: Save checkpoint every N codes processed
            quality_validator: Optional quality validator to filter examples

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
        if checkpoint_interval <= 0:
            raise ValueError(f"checkpoint_interval must be positive, got {checkpoint_interval}")

        # Load existing dataset from checkpoint if available
        if checkpoint_manager:
            dataset = checkpoint_manager.dataset.copy()
            # Filter out codes that are already processed
            codes_to_process = [
                code for code in codes if not checkpoint_manager.is_code_processed(code.code)
            ]
            if codes_to_process:
                logger.info(
                    f"Resuming from checkpoint: {len(codes) - len(codes_to_process)} "
                    f"codes already processed, {len(codes_to_process)} remaining"
                )
            codes = codes_to_process
        else:
            dataset = []

        failed_codes = []
        codes_processed_since_checkpoint = 0

        for i, code in enumerate(codes):
            try:
                examples = self.generate_for_code(
                    code, num_examples=examples_per_code, quality_validator=quality_validator
                )

                for example in examples:
                    example_data = {
                        "text": example,
                        "label": code.code,
                        "description": code.description,
                        "category": code.category,
                        "level": code.level,
                    }
                    dataset.append(example_data)

                # Mark code as processed in checkpoint
                if checkpoint_manager:
                    checkpoint_manager.mark_code_processed(code.code)
                    checkpoint_manager.add_examples(
                        [
                            example_data
                            for example_data in dataset[-len(examples) :]  # Only new examples
                        ]
                    )
                    codes_processed_since_checkpoint += 1

            except SyntheticDataGenerationError as e:
                logger.error(f"Failed to generate examples for {code.code}: {e}")
                failed_codes.append(code.code)

                # Mark as failed in checkpoint
                if checkpoint_manager:
                    checkpoint_manager.mark_code_failed(code.code)

                if not continue_on_error:
                    raise

            # Save checkpoint periodically
            if checkpoint_manager and codes_processed_since_checkpoint >= checkpoint_interval:
                checkpoint_manager.save()
                codes_processed_since_checkpoint = 0

            if (i + 1) % batch_size == 0:
                logger.info(
                    f"Progress: {i + 1}/{len(codes)} codes processed, "
                    f"{len(dataset)} total examples, {len(failed_codes)} failures"
                )

        # Save final checkpoint
        if checkpoint_manager:
            checkpoint_manager.save()

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
