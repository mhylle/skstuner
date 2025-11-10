"""Model creation and configuration for SKS classification."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)


def load_model_config(config_path: Path) -> Dict[str, Any]:
    """
    Load model configuration from YAML file.

    Args:
        config_path: Path to model config YAML file

    Returns:
        Dictionary with model configuration
    """
    logger.info(f"Loading model config from {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config for model: {config.get('model_name')}")
    return config


def create_model(
    config: Dict[str, Any],
    num_labels: int,
    cache_dir: Optional[Path] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Create model and tokenizer for SKS classification.

    Args:
        config: Model configuration dictionary
        num_labels: Number of classification labels
        cache_dir: Optional cache directory for model files

    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = config["model_name"]
    model_type = config.get("model_type", "encoder")
    use_lora = config.get("use_lora", False)

    logger.info(f"Creating {model_type} model: {model_name}")
    logger.info(f"Number of labels: {num_labels}")
    logger.info(f"Using LoRA: {use_lora}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load model configuration
    model_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # For decoder models, set pad_token_id
    if model_type == "decoder":
        model_config.pad_token_id = tokenizer.pad_token_id

    # Load base model
    logger.info("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=model_config,
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # Resize token embeddings if we added tokens
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA if specified
    if use_lora:
        logger.info("Applying LoRA configuration...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=_get_lora_target_modules(model_name, model_type),
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(
        f"Trainable %: {100 * trainable_params / total_params:.2f}%"
    )

    return model, tokenizer


def _get_lora_target_modules(model_name: str, model_type: str) -> list[str]:
    """
    Get target modules for LoRA based on model architecture.

    Args:
        model_name: Name of the model
        model_type: Type of model (encoder or decoder)

    Returns:
        List of module names to apply LoRA to
    """
    model_name_lower = model_name.lower()

    # XLM-RoBERTa and similar encoder models
    if "xlm" in model_name_lower or "roberta" in model_name_lower or "bert" in model_name_lower:
        return ["query", "value"]

    # Phi-3 models
    elif "phi" in model_name_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Gemma models
    elif "gemma" in model_name_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    # LLaMA and similar models
    elif "llama" in model_name_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Default for decoder models
    elif model_type == "decoder":
        return ["q_proj", "v_proj"]

    # Default for encoder models
    else:
        return ["query", "value"]


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: Path,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
) -> None:
    """
    Save trained model, tokenizer, and label mappings.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_dir: Directory to save model
        label_to_id: Label to ID mapping
        id_to_label: ID to label mapping
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {output_dir}")

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mappings
    label_mapping_path = output_dir / "label_mapping.json"
    import json

    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_to_id": label_to_id,
                "id_to_label": {str(k): v for k, v in id_to_label.items()},
                "num_labels": len(label_to_id),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"Model saved successfully to {output_dir}")


def load_trained_model(
    model_dir: Path,
) -> tuple[PreTrainedModel, PreTrainedTokenizer, Dict[str, int], Dict[int, str]]:
    """
    Load a trained model with its tokenizer and label mappings.

    Args:
        model_dir: Directory containing the saved model

    Returns:
        Tuple of (model, tokenizer, label_to_id, id_to_label)
    """
    model_dir = Path(model_dir)
    logger.info(f"Loading trained model from {model_dir}")

    # Load label mappings
    label_mapping_path = model_dir / "label_mapping.json"
    import json

    with open(label_mapping_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    label_to_id = label_data["label_to_id"]
    id_to_label = {int(k): v for k, v in label_data["id_to_label"].items()}

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    logger.info(f"Loaded model with {label_data['num_labels']} labels")

    return model, tokenizer, label_to_id, id_to_label
