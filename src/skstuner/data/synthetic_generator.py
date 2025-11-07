"""Synthetic clinical note generation using LLMs"""
from typing import List, Optional
import logging
import re
from anthropic import Anthropic
from openai import AzureOpenAI
from skstuner.data.sks_parser import SKSCode
from skstuner.data.prompt_templates import PromptTemplateManager

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic clinical notes using Claude or Azure OpenAI"""

    MIN_NOTE_LENGTH = 20  # Minimum characters to filter out incomplete/empty notes

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_api_version: str = "2024-02-15-preview"
    ):
        """
        Initialize the synthetic data generator

        Args:
            api_key: API key for the provider (Anthropic or Azure OpenAI)
            model: Model name to use
            provider: Either "anthropic" or "azure" (default: "anthropic")
            azure_endpoint: Azure OpenAI endpoint URL (required if provider="azure")
            azure_deployment: Azure OpenAI deployment name (required if provider="azure")
            azure_api_version: Azure OpenAI API version (default: "2024-02-15-preview")
        """
        self.provider = provider.lower()
        self.model = model
        self.template_manager = PromptTemplateManager()

        if self.provider == "azure":
            if not azure_endpoint or not azure_deployment:
                raise ValueError("azure_endpoint and azure_deployment are required when using Azure provider")
            if not api_key:
                raise ValueError("api_key is required for Azure OpenAI")

            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint
            )
            self.deployment_name = azure_deployment
            logger.info(f"Initialized Azure OpenAI client with deployment: {azure_deployment}")

        elif self.provider == "anthropic":
            if not api_key:
                raise ValueError("api_key is required for Anthropic")

            self.client = Anthropic(api_key=api_key)
            logger.info(f"Initialized Anthropic client with model: {model}")

        else:
            raise ValueError(f"Unknown provider: {provider}. Must be 'anthropic' or 'azure'")

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

        # Call LLM API based on provider
        try:
            if self.provider == "azure":
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7
                )

                if not response.choices or not response.choices[0].message.content:
                    logger.error(f"Unexpected API response format for {code.code}")
                    return []

                response_text = response.choices[0].message.content

            elif self.provider == "anthropic":
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

            else:
                logger.error(f"Unknown provider: {self.provider}")
                return []

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
