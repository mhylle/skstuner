"""Dataset preparation for SKS code classification."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class SKSDataset:
    """Dataset handler for SKS code classification."""

    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        label_to_id: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize SKS dataset.

        Args:
            data_path: Path to synthetic data JSON file
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            label_to_id: Mapping from SKS codes to label IDs
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id or {}

        # Load and validate data
        self.examples = self._load_data()
        if not self.label_to_id:
            self.label_to_id = self._build_label_mapping()

        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.num_labels = len(self.label_to_id)

        logger.info(f"Loaded {len(self.examples)} examples with {self.num_labels} unique labels")

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load synthetic data from JSON file."""
        logger.info(f"Loading data from {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list format and dict with 'examples' key
        if isinstance(data, dict) and "examples" in data:
            examples = data["examples"]
        elif isinstance(data, list):
            examples = data
        else:
            raise ValueError(f"Unexpected data format in {self.data_path}")

        logger.info(f"Loaded {len(examples)} examples")
        return examples

    def _build_label_mapping(self) -> Dict[str, int]:
        """Build mapping from SKS codes to integer labels."""
        all_codes = set()
        for example in self.examples:
            code = example.get("sks_code") or example.get("code")
            if code:
                all_codes.add(code)

        label_to_id = {code: idx for idx, code in enumerate(sorted(all_codes))}
        logger.info(f"Built label mapping with {len(label_to_id)} unique codes")
        return label_to_id

    def to_dataframe(self) -> pd.DataFrame:
        """Convert examples to pandas DataFrame."""
        records = []
        for example in self.examples:
            text = example.get("text") or example.get("clinical_note", "")
            code = example.get("sks_code") or example.get("code")

            if text and code:
                records.append(
                    {
                        "text": text,
                        "sks_code": code,
                        "label_id": self.label_to_id.get(code, -1),
                        "description": example.get("description", ""),
                        "category": example.get("category", ""),
                    }
                )

        df = pd.DataFrame(records)
        # Remove any examples with invalid labels
        df = df[df["label_id"] >= 0]
        logger.info(f"Created DataFrame with {len(df)} valid examples")
        return df

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset."""
        df = self.to_dataframe()

        dataset = Dataset.from_pandas(df)

        # Apply tokenization
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            # Add labels
            tokenized["labels"] = examples["label_id"]
            return tokenized

        dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        logger.info(f"Created HuggingFace dataset with {len(dataset)} examples")
        return dataset


def prepare_datasets(
    data_path: Path,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    label_to_id: Optional[Dict[str, int]] = None,
) -> Tuple[DatasetDict, Dict[str, int], Dict[int, str]]:
    """
    Prepare train/val/test datasets for SKS classification.

    Args:
        data_path: Path to synthetic data JSON file
        tokenizer: Tokenizer for text processing
        max_length: Maximum sequence length
        test_size: Fraction of data for test set
        val_size: Fraction of train data for validation set
        random_state: Random seed for reproducibility
        label_to_id: Optional pre-built label mapping

    Returns:
        Tuple of (DatasetDict, label_to_id, id_to_label)
    """
    logger.info("Preparing datasets...")

    # Load data
    sks_dataset = SKSDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        label_to_id=label_to_id,
    )

    # Get DataFrame
    df = sks_dataset.to_dataframe()

    # Check if we have enough samples for stratified splitting
    # Need at least 2 samples per class for stratification
    min_samples_per_class = df["label_id"].value_counts().min()
    use_stratify = min_samples_per_class >= 2

    if not use_stratify:
        logger.warning(
            f"Insufficient samples per class (min={min_samples_per_class}) for stratified splitting. "
            "Using random split instead."
        )

    # Split into train+val and test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_id"] if use_stratify else None
    )

    # Check again for the second split
    min_samples_train_val = train_val_df["label_id"].value_counts().min()
    use_stratify_val = min_samples_train_val >= 2

    if not use_stratify_val:
        logger.warning(
            f"Insufficient samples per class (min={min_samples_train_val}) in train+val for stratified splitting. "
            "Using random split instead."
        )

    # Split train+val into train and val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_df["label_id"] if use_stratify_val else None,
    )

    logger.info(
        f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # Create tokenization function
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = examples["label_id"]
        return tokenized

    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    # Tokenize
    train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Create DatasetDict
    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        }
    )

    logger.info("Dataset preparation complete")

    return dataset_dict, sks_dataset.label_to_id, sks_dataset.id_to_label
