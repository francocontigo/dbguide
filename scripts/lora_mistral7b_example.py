"""Simple LoRA training script on top of Mistral 7B Instruct.

Requirements (already listed in pyproject):
- transformers
- datasets
- peft
- accelerate
- bitsandbytes (for 4/8-bit quantization, optional but recommended)

IMPORTANT:
- Training Mistral 7B requires a GPU with substantial VRAM. On weaker
	machines, reduce `per_device_train_batch_size` or use a smaller model.
- This script is a starting point; adjust hyperparameters for your use case.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import datasets
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	DataCollatorForLanguageModeling,
	Trainer,
	TrainingArguments,
)


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


@dataclass
class LoraTrainConfig:
	base_model: str = BASE_MODEL
	data_path: Path = Path("data/lora_training/train.jsonl")
	output_dir: Path = Path("data/lora_training/out_mistral7b_lora")
	num_train_epochs: int = 1
	learning_rate: float = 2e-4
	per_device_train_batch_size: int = 1
	gradient_accumulation_steps: int = 4
	lora_r: int = 8
	lora_alpha: int = 16
	lora_dropout: float = 0.05


def load_jsonl_dataset(path: Path) -> Dataset:
	"""Load a local JSONL dataset.

	Expected format for each JSON line:
		{"instruction": "natural language question", "response": "target SQL"}
	"""

	if not path.exists():
		raise SystemExit(
			f"Arquivo de treino não encontrado em {path}. "
			"Crie um JSONL com campos 'instruction' e 'response'."
		)

	ds = datasets.load_dataset("json", data_files=str(path), split="train")
	return ds


def build_prompt(example: Dict[str, str]) -> str:
	"""Build an instruction-style prompt for the model."""

	instr = example.get("instruction", "").strip()
	resp = example.get("response", "").strip()

	# Simple instruction-style chat format
	prompt = (
		"<s>[INST] Você é um assistente especializado em SQL. "
		"Gere apenas uma query SQL de leitura (SELECT) para o pedido abaixo.\n\n"
		f"Pedido: {instr} [/INST]\n"
		f"{resp}</s>"
	)
	return prompt


def tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
	def _map_fn(example: Dict[str, str]) -> Dict[str, List[int]]:
		text = build_prompt(example)
		out = tokenizer(text, truncation=True, max_length=1024)
		return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

	return ds.map(_map_fn, remove_columns=ds.column_names)


def train_lora(cfg: LoraTrainConfig) -> None:
	cfg.output_dir.mkdir(parents=True, exist_ok=True)

	print(f"Loading dataset from {cfg.data_path}...")
	raw_ds = load_jsonl_dataset(cfg.data_path)

	print(f"Loading tokenizer and base model: {cfg.base_model}...")
	tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	model = AutoModelForCausalLM.from_pretrained(
		cfg.base_model,
		load_in_8bit=True,
		device_map="auto",
	)

	print("Applying LoRA configuration...")
	lora_cfg = LoraConfig(
		r=cfg.lora_r,
		lora_alpha=cfg.lora_alpha,
		lora_dropout=cfg.lora_dropout,
		bias="none",
		task_type="CAUSAL_LM",
	)
	model = get_peft_model(model, lora_cfg)

	print("Tokenizing dataset...")
	tokenized_ds = tokenize_dataset(raw_ds, tokenizer)

	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

	training_args = TrainingArguments(
		output_dir=str(cfg.output_dir),
		num_train_epochs=cfg.num_train_epochs,
		learning_rate=cfg.learning_rate,
		per_device_train_batch_size=cfg.per_device_train_batch_size,
		gradient_accumulation_steps=cfg.gradient_accumulation_steps,
		logging_steps=10,
		save_steps=200,
		save_total_limit=2,
		optim="paged_adamw_8bit",
		fp16=True,
		report_to=[],
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_ds,
		data_collator=data_collator,
	)

	print("Starting LoRA training...")
	trainer.train()

	print("Saving LoRA adapter...")
	trainer.save_model()
	tokenizer.save_pretrained(str(cfg.output_dir))

	print("Training complete. To use it in dbguide:")
	print("- Deploy this model/adapter to a server (e.g., Ollama, vLLM, etc.)")
	print("- Configure OLLAMA_MODEL_MYSQL/REDSHIFT or OPENAI_MODEL_* with its name/ID.")


def parse_args() -> LoraTrainConfig:
	parser = argparse.ArgumentParser(description="Train a LoRA adapter on Mistral 7B for SQL.")
	parser.add_argument("--data-path", type=str, default="data/lora_training/train.jsonl")
	parser.add_argument("--output-dir", type=str, default="data/lora_training/out_mistral7b_lora")
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--grad-accum", type=int, default=4)

	args = parser.parse_args()
	return LoraTrainConfig(
		data_path=Path(args.data_path),
		output_dir=Path(args.output_dir),
		num_train_epochs=args.epochs,
		learning_rate=args.lr,
		per_device_train_batch_size=args.batch_size,
		gradient_accumulation_steps=args.grad_accum,
	)


def main() -> None:
	cfg = parse_args()
	train_lora(cfg)


if __name__ == "__main__":
	main()
