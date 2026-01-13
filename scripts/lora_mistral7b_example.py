"""Professional LoRA training script for SQL generation using Mistral 7B Instruct.

This script fine-tunes Mistral 7B with LoRA (Low-Rank Adaptation) for SQL generation
tasks. It uses parameter-efficient fine-tuning to adapt the model with minimal resources.

Requirements (already listed in pyproject.toml):
    - transformers: Model loading and training
    - datasets: Dataset handling
    - peft: LoRA implementation
    - accelerate: Distributed training support
    - bitsandbytes: 4/8-bit quantization (optional but recommended)

Hardware Requirements:
    - GPU with 16GB+ VRAM (24GB recommended for batch_size > 1)
    - For lower VRAM: reduce batch_size or use smaller models (e.g., Mistral 7B)
    - CPU training is possible but extremely slow (not recommended)

Usage:
    # Basic usage with default parameters
    uv run python scripts/lora_mistral7b_example.py

    # Custom dataset and output directory
    uv run python scripts/lora_mistral7b_example.py \
        --data-path data/custom_train.jsonl \
        --output-dir data/lora_output

    # Adjust training parameters
    uv run python scripts/lora_mistral7b_example.py \
        --epochs 3 \
        --lr 1e-4 \
        --batch-size 2 \
        --grad-accum 8

Output:
    The script saves the trained LoRA adapter to the specified output directory.
    To use it:
    1. Deploy to inference server (Ollama, vLLM, TGI, etc.)
    2. Configure environment variables in .env:
       OLLAMA_MODEL_MYSQL=mistral:7b-instruct-sql
       OLLAMA_MODEL_REDSHIFT=mistral:7b-instruct-sql

Author: DBGuide Team
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import datasets
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Model configuration
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_MAX_LENGTH = 1024


@dataclass
class LoraTrainConfig:
    """Configuration for LoRA training.

    Attributes:
        base_model: HuggingFace model ID to fine-tune
        data_path: Path to training dataset (JSONL format)
        output_dir: Directory to save LoRA adapter and checkpoints
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        per_device_train_batch_size: Batch size per GPU/CPU
        gradient_accumulation_steps: Steps to accumulate gradients before update
        max_length: Maximum sequence length for tokenization
        lora_r: LoRA attention dimension (rank)
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout probability for LoRA layers
        load_in_8bit: Use 8-bit quantization for base model
        use_fp16: Use mixed precision training (FP16)
        save_steps: Save checkpoint every N steps
        logging_steps: Log training metrics every N steps
    """
    base_model: str = BASE_MODEL
    data_path: Path = Path("data/lora_training/train.jsonl")
    output_dir: Path = Path("data/lora_training/out_mistral7b_lora")
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_length: int = DEFAULT_MAX_LENGTH
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    load_in_8bit: bool = True
    use_fp16: bool = True
    save_steps: int = 200
    logging_steps: int = 10
    save_total_limit: int = 2

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
            FileNotFoundError: If data_path doesn't exist
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {self.data_path}. "
                "Create a JSONL file with 'instruction' and 'response' fields."
            )

        if self.num_train_epochs < 1:
            raise ValueError("num_train_epochs must be >= 1")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        if self.per_device_train_batch_size < 1:
            raise ValueError("per_device_train_batch_size must be >= 1")

        if self.lora_r < 1 or self.lora_r > 64:
            raise ValueError("lora_r must be between 1 and 64")

        if self.lora_alpha < 1:
            raise ValueError("lora_alpha must be >= 1")

        logger.info("Configuration validated successfully")


def load_jsonl_dataset(path: Path) -> Dataset:
    """Load training dataset from JSONL file.

    Expected format for each JSON line:
        {
            "instruction": "Natural language SQL question",
            "response": "Target SQL query"
        }

    Example:
        {"instruction": "Show all users from last month",
         "response": "SELECT * FROM users WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 MONTH);"}

    Args:
        path: Path to JSONL file containing training examples

    Returns:
        Dataset object ready for processing

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If dataset is empty or has invalid format
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            "Create a JSONL file with 'instruction' and 'response' fields."
        )

    logger.info(f"Loading dataset from {path}...")
    ds = datasets.load_dataset("json", data_files=str(path), split="train")

    if len(ds) == 0:
        raise ValueError(f"Dataset is empty: {path}")

    # Validate required fields
    required_fields = {"instruction", "response"}
    if not required_fields.issubset(ds.column_names):
        raise ValueError(
            f"Dataset must contain {required_fields} fields. "
            f"Found: {ds.column_names}"
        )

    logger.info(f"Loaded {len(ds)} training examples")
    return ds


def build_prompt(example: Dict[str, str], dialect: str = "generic") -> str:
    """Build instruction-style prompt for SQL generation.

    Uses Mistral's chat template format: <s>[INST] instruction [/INST] response</s>

    Args:
        example: Dict with 'instruction' and 'response' keys
        dialect: SQL dialect hint (e.g., 'mysql', 'redshift', 'generic')

    Returns:
        Formatted prompt string ready for tokenization

    Example:
        >>> example = {
        ...     "instruction": "Show all users",
        ...     "response": "SELECT * FROM users;"
        ... }
        >>> prompt = build_prompt(example, dialect="mysql")
    """
    instr = example.get("instruction", "").strip()
    resp = example.get("response", "").strip()

    if not instr or not resp:
        logger.warning(f"Empty instruction or response in example: {example}")

    # Mistral instruction format with SQL-specific system message
    system_msg = (
        "You are a SQL expert assistant. "
        "Generate safe, read-only SQL queries (SELECT) for the given request. "
        f"Target dialect: {dialect.upper()}."
    )

    prompt = (
        f"<s>[INST] {system_msg}\n\n"
        f"Request: {instr} [/INST]\n"
        f"{resp}</s>"
    )
    return prompt


def tokenize_dataset(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = DEFAULT_MAX_LENGTH
) -> Dataset:
    """Tokenize dataset for causal language modeling.

    Converts each example to tokenized format with input_ids and attention_mask.
    Removes original columns to save memory.

    Args:
        ds: Raw dataset with 'instruction' and 'response' fields
        tokenizer: Pre-trained tokenizer
        max_length: Maximum sequence length (longer sequences are truncated)

    Returns:
        Tokenized dataset ready for training
    """
    def _map_fn(example: Dict[str, str]) -> Dict[str, List[int]]:
        """Tokenize single example."""
        text = build_prompt(example)
        out = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False  # Dynamic padding in collator
        )
        return {
            "input_ids": out["input_ids"],
            "attention_mask": out["attention_mask"]
        }

    logger.info("Tokenizing dataset...")
    tokenized = ds.map(
        _map_fn,
        remove_columns=ds.column_names,
        desc="Tokenizing examples"
    )

    # Log statistics
    lengths = [len(ids) for ids in tokenized["input_ids"]]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    logger.info(f"Tokenization complete. Avg length: {avg_len:.1f}, Max: {max_len}")

    return tokenized


def train_lora(cfg: LoraTrainConfig) -> None:
    """Execute LoRA training pipeline.

    Pipeline steps:
    1. Validate configuration
    2. Load and validate dataset
    3. Load tokenizer and base model
    4. Apply LoRA configuration
    5. Tokenize dataset
    6. Train model
    7. Save adapter and tokenizer

    Args:
        cfg: Training configuration

    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If training fails
    """
    logger.info("=" * 70)
    logger.info("Starting LoRA Training Pipeline")
    logger.info("=" * 70)

    # Validate configuration
    cfg.validate()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    raw_ds = load_jsonl_dataset(cfg.data_path)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {cfg.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Load base model with quantization
    logger.info(f"Loading base model: {cfg.base_model}")
    logger.info(f"Using 8-bit quantization: {cfg.load_in_8bit}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        load_in_8bit=cfg.load_in_8bit,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    logger.info("Applying LoRA configuration")
    logger.info(f"  - LoRA rank (r): {cfg.lora_r}")
    logger.info(f"  - LoRA alpha: {cfg.lora_alpha}")
    logger.info(f"  - LoRA dropout: {cfg.lora_dropout}")

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Mistral attention modules
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Tokenize dataset
    tokenized_ds = tokenize_dataset(raw_ds, tokenizer, cfg.max_length)

    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM (not masked)
    )

    # Training arguments
    logger.info("Configuring training arguments")
    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        optim="paged_adamw_8bit" if cfg.load_in_8bit else "adamw_torch",
        fp16=cfg.use_fp16,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        report_to=[],  # Disable wandb/tensorboard
        logging_first_step=True,
        save_safetensors=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    # Train
    logger.info("=" * 70)
    logger.info("Starting LoRA training")
    logger.info("=" * 70)

    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise RuntimeError(f"Training failed: {e}") from e

    # Save adapter
    logger.info("Saving LoRA adapter and tokenizer")
    trainer.save_model()
    tokenizer.save_pretrained(str(cfg.output_dir))

    # Success message
    logger.info("=" * 70)
    logger.info("✅ Training completed successfully!")
    logger.info("=" * 70)
    logger.info(f"Output directory: {cfg.output_dir}")
    logger.info("")
    logger.info("Next steps to use this adapter:")
    logger.info("")
    logger.info("1. Deploy to inference server:")
    logger.info("   - Ollama: Create Modelfile and run 'ollama create'")
    logger.info("   - vLLM: Use --lora-modules flag")
    logger.info("   - TGI: Use --lora-adapters flag")
    logger.info("")
    logger.info("2. Configure DBGuide (.env file):")
    logger.info("   OLLAMA_MODEL_MYSQL=mistral-sql")
    logger.info("   OLLAMA_MODEL_REDSHIFT=mistral-sql")
    logger.info("")
    logger.info("3. Run DBGuide:")
    logger.info("   uv run python run_app.py")
    logger.info("=" * 70)


def parse_args() -> LoraTrainConfig:
    """Parse command-line arguments.

    Returns:
        Configured LoraTrainConfig instance
    """
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter on Mistral 7B for SQL generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/lora_training/train.jsonl",
        help="Path to training dataset (JSONL format)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/lora_training/out_mistral7b_lora",
        help="Output directory for LoRA adapter"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum sequence length"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA attention dimension (rank)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA scaling parameter"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability"
    )

    # Model arguments
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL,
        help="Base model HuggingFace ID"
    )
    parser.add_argument(
        "--no-8bit",
        action="store_true",
        help="Disable 8-bit quantization (requires more VRAM)"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 mixed precision training"
    )

    args = parser.parse_args()

    return LoraTrainConfig(
        base_model=args.base_model,
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_8bit=not args.no_8bit,
        use_fp16=not args.no_fp16,
    )


def main() -> None:
    """Main entry point for LoRA training script."""
    try:
        cfg = parse_args()
        train_lora(cfg)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
