"""Script simples para treino LoRA em cima do Mistral 7B Instruct.

Requisitos (pyproject já atualizado):
- transformers
- datasets
- peft
- accelerate
- bitsandbytes (para quantização 4/8 bits, opcional mas recomendado)

IMPORTANTE:
- Treinar Mistral 7B exige GPU com bastante VRAM. Em máquina fraca, reduza o
  `per_device_train_batch_size` ou use um modelo menor.
- Este script é um ponto de partida; ajuste hiperparâmetros para seu caso.
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
	"""Carrega um dataset JSONL local.

	Formato esperado de cada linha JSON:
		{"instruction": "pergunta em linguagem natural", "response": "SQL alvo"}
	"""

	if not path.exists():
		raise SystemExit(
			f"Arquivo de treino não encontrado em {path}. "
			"Crie um JSONL com campos 'instruction' e 'response'."
		)

	ds = datasets.load_dataset("json", data_files=str(path), split="train")
	return ds


def build_prompt(example: Dict[str, str]) -> str:
	"""Monta um prompt estilo instrução → resposta.

	Você pode customizar para refletir o estilo do DBGuide (EXPLICACAO/CHECKS etc.).
	"""

	instr = example.get("instruction", "").strip()
	resp = example.get("response", "").strip()

	# Formato simples estilo chat de instrução
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

	print(f"Carregando dataset de {cfg.data_path}...")
	raw_ds = load_jsonl_dataset(cfg.data_path)

	print(f"Carregando tokenizer e modelo base: {cfg.base_model}...")
	tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	model = AutoModelForCausalLM.from_pretrained(
		cfg.base_model,
		load_in_8bit=True,
		device_map="auto",
	)

	print("Aplicando configuração LoRA...")
	lora_cfg = LoraConfig(
		r=cfg.lora_r,
		lora_alpha=cfg.lora_alpha,
		lora_dropout=cfg.lora_dropout,
		bias="none",
		task_type="CAUSAL_LM",
	)
	model = get_peft_model(model, lora_cfg)

	print("Tokenizando dataset...")
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

	print("Iniciando treino LoRA...")
	trainer.train()

	print("Salvando adapter LoRA...")
	trainer.save_model()
	tokenizer.save_pretrained(str(cfg.output_dir))

	print("Treino concluído. Para usar no dbguide:")
	print("- Suba esse modelo/adapter em um servidor (ex.: Ollama, vLLM, etc.)")
	print("- Configure OLLAMA_MODEL_MYSQL/REDSHIFT ou OPENAI_MODEL_* com o nome/ID dele.")


def parse_args() -> LoraTrainConfig:
	parser = argparse.ArgumentParser(description="Treinar LoRA em Mistral 7B para SQL.")
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
