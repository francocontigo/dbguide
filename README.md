
# DBGuide

Gerador de SQL seguro usando RAG (recuperação de conhecimento em Markdown) e modelos de linguagem (Ollama ou OpenAI).

## Sobre o projeto

O DBGuide segue boas práticas de Python:
- **Tipagem estática** em todas as funções e dataclasses.
- **Docstrings** e comentários explicativos em todos os módulos.
- Estrutura modular: fácil de estender e manter.

### Principais módulos
- `dbguide/app/retrieval.py`: leitura, indexação e busca híbrida nos cards.
- `dbguide/app/llm.py`: wrappers para modelos Ollama e OpenAI.
- `dbguide/app/prompts.py`: geração de prompts e compactação de cards.
- `dbguide/app/safety.py`: checagem de segurança e parsing de respostas.
- `dbguide/ingest/build_index.py`: script para indexar os cards.

### Exemplo de uso programático
```python
from dbguide.app.retrieval import read_markdown_docs, build_vector_index, build_bm25, hybrid_search
docs = read_markdown_docs("dbguide/corpus")
col = build_vector_index(docs)
bm25 = build_bm25(docs)
results = hybrid_search("minha pergunta", col, bm25, docs)
```

### Contribuição
- Siga o padrão de tipagem e docstrings.
- Comente trechos complexos.
- Sugestões e PRs são bem-vindos!

1. Instalação
--------------

- Pré‑requisitos:
	- Python (o projeto usa `uv` para gerenciar deps).
- Na raiz do projeto:

```bash
uv sync
```

2. Configuração (.env)
----------------------

Crie um arquivo `.env` na raiz. Exemplo mínimo:

```bash
# Ollama (local)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL_MYSQL=mistral:7b-instruct
OLLAMA_MODEL_REDSHIFT=mistral:7b-instruct

# OpenAI (opcional)
OPENAI_API_KEY=sua_chave_aqui
OPENAI_MODEL_MYSQL=gpt-4o-mini
OPENAI_MODEL_REDSHIFT=gpt-4o-mini
```

3. Construir o índice RAG
-------------------------

Os cards são arquivos `.md` em `dbguide/corpus/` (por exemplo, `pattern_cards/` e `query_cards/`).
Para indexar os cards no Chroma + BM25, rode na raiz do projeto:

```bash
uv run python -m dbguide.ingest.build_index
```

Isso gera:

- `data/chroma/` (índice vetorial)
- `data/bm25.pkl` (índice BM25)

4. Rodar o chat
----------------

Ainda na raiz do projeto:

```bash
uv run streamlit run dbguide/app/streamlit_app.py
```

Na interface você escolhe:

- Dialeto (MySQL ou Redshift).
- Provedor (Ollama ou OpenAI).
- Faz perguntas em linguagem natural, e o app devolve:
	- SQL sugerido.
	- Explicação em bullets.
	- Checks para validar duplicidade/volume.

5. Como o RAG funciona (resumo)
--------------------------------

- Os `.md` em `dbguide/corpus/` são lidos e indexados em:
	- ChromaDB (vetorial, via Sentence Transformers).
	- BM25 (keyword).
- Na pergunta, o app:
	- Recupera os cards mais relevantes (híbrido: vetorial + BM25).
	- Monta um prompt com regras de segurança + cards.
	- Chama o modelo (Ollama ou OpenAI) para gerar o SQL.

6. LoRA / modelos fine‑tunados
------------------------------

O projeto **não treina LoRA** diretamente; ele apenas consome o modelo cujo nome você colocar nas variáveis de ambiente.

- Para Ollama: defina `OLLAMA_MODEL_MYSQL` e `OLLAMA_MODEL_REDSHIFT` com o nome do modelo (por exemplo, `mistral:7b-instruct-dbguide-lora`).
- Para OpenAI: use o ID de um modelo fine‑tunado nas variáveis `OPENAI_MODEL_MYSQL` / `OPENAI_MODEL_REDSHIFT`.

Veja também `scripts/lora_mistral7b_example.py` como esqueleto de treino LoRA offline.

