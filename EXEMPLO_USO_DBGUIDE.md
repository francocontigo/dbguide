# Exemplo de uso do dbguide

## 1. Preparar o ambiente

1. Instale dependencias usando uv na raiz do projeto:

```bash
uv sync
```

2. Configure o arquivo `.env` na raiz. Exemplo minimo:

```bash
# Ollama (local)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL_MYSQL=mistral:7b-instruct
OLLAMA_MODEL_REDSHIFT=mistral:7b-instruct

# OpenAI (opcional)
OPENAI_API_KEY=coloque_sua_chave_aqui
OPENAI_MODEL_MYSQL=gpt-4o-mini
OPENAI_MODEL_REDSHIFT=gpt-4o-mini
```

## 2. Construir o indice RAG

Na raiz do projeto:

```bash
uv run python -m dbguide.ingest.build_index
```

Isso vai ler os arquivos `.md` em `dbguide/corpus/` e gerar:

- `data/chroma/`
- `data/bm25.pkl`

## 3. Rodar a interface de chat

Ainda na raiz do projeto:

```bash
uv run streamlit run dbguide/app/streamlit_app.py
```

Na UI voce podera:

- Escolher o dialeto (MySQL ou Redshift).
- Escolher o provedor (Ollama ou OpenAI).
- Conversar em linguagem natural para gerar SQL seguro, com RAG usando os cards em `dbguide/corpus/`.

## 4. Exemplo de pedido

> Quero a taxa de acordos por semana por produto, no ultimo mes, filtrando pela empresa 7772.

A partir desse pedido, o dbguide vai recuperar os cards mais relevantes, montar o contexto, chamar o LLM e retornar:

- A query SQL.
- Explicacao em bullets.
- Checks sugeridos para validar duplicidade/volume.

## 5. Usar modelos LoRA ou fine-tunados

O dbguide nao faz treino ou fine-tuning dentro do codigo. Ele **apenas consome** o modelo cujo nome voce informar nas variaveis de ambiente:

- Para Ollama: `OLLAMA_MODEL_MYSQL` e `OLLAMA_MODEL_REDSHIFT`.
- Para OpenAI: `OPENAI_MODEL_MYSQL` e `OPENAI_MODEL_REDSHIFT`.

Se voce tiver um modelo LoRA ou fine-tunado:

- **Ollama**: carregue o modelo (por exemplo, um modelo LoRA) no Ollama e use exatamente o nome dele nessas variaveis. O dbguide vai so chamar `model=<seu_modelo>` pela API do Ollama.
- **OpenAI**: crie um modelo fine-tunado na conta OpenAI, copie o ID do modelo e coloque nas variaveis `OPENAI_MODEL_MYSQL` / `OPENAI_MODEL_REDSHIFT`. O codigo ja envia o `model` que voce configurar.

Assim, toda a logica de RAG e prompts permanece igual, apenas mudando qual modelo e chamado (base, LoRA ou fine-tunado).
