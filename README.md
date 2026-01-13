# DBGuide

Safe SQL generator using RAG (Retrieval-Augmented Generation with Markdown knowledge) and LLMs (Ollama or OpenAI).

## About the Project

DBGuide follows Python best practices with a clean, modular architecture:
- ✅ **SOLID principles** applied throughout
- ✅ **Dependency injection** for flexibility and testability
- ✅ **Type annotations** on all functions and classes
- ✅ **Clean architecture** with 4 layers (Domain, Models, Services, Application)

### Architecture Overview

```
┌─────────────────────────────────────┐
│     Application Layer               │  Streamlit UI
├─────────────────────────────────────┤
│     Services Layer                  │  Business Logic
├─────────────────────────────────────┤
│     Domain Layer                    │  Interfaces
├─────────────────────────────────────┤
│     Models Layer                    │  Data Models
└─────────────────────────────────────┘
```

### Key Features

- ✅ **Hybrid Search**: Combines vector similarity (ChromaDB) with keyword search (BM25)
- ✅ **Intelligent Filtering**: LLM-powered metadata filtering for query routing
- ✅ **Multi-LLM Support**: Works with Ollama (local) or OpenAI (cloud)
- ✅ **SQL Validation**: Built-in SQL parsing and safety checks
- ✅ **Clean Architecture**: SOLID principles with dependency injection

### Main Modules

**Domain Layer:**
- `dbguide/domain/interfaces.py` - Abstract interfaces (LLMProvider, RetrievalService, MetadataFilterService, etc.)

**Models Layer:**
- `dbguide/models/document.py` - Document data model with metadata

**Services Layer:**
- `dbguide/services/llm_providers.py` - LLM implementations (Ollama, OpenAI)
- `dbguide/services/retrieval_service.py` - Hybrid search (Vector + BM25)
- `dbguide/services/metadata_filter.py` - Intelligent query routing
- `dbguide/services/sql_validator.py` - SQL validation and parsing
- `dbguide/services/prompt_builder.py` - Prompt construction
- `dbguide/services/document_loader.py` - Document loading with YAML frontmatter
- `dbguide/services/indexing.py` - Vector and BM25 index building

---

## Example Usage

### Basic SQL Generation Pipeline

```python
from dbguide.services.llm_providers import OllamaProvider
from dbguide.services.retrieval_service import HybridRetrievalService
from dbguide.services.prompt_builder import SQLPromptBuilder
from dbguide.services.sql_validator import BasicSQLValidator, SQLOutputParser
from dbguide.services.indexing import BM25IndexBuilder
import chromadb

# 1. Setup
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_collection(name="sql_cards")
bm25_index, documents = BM25IndexBuilder.load_index("data/bm25.pkl")

# 2. Services
retrieval = HybridRetrievalService(collection, bm25_index, documents)
prompt_builder = SQLPromptBuilder()
llm = OllamaProvider()
validator = BasicSQLValidator()
parser = SQLOutputParser()

# 3. User question
question = "Show me all users who registered in the last 30 days"

# 4. Retrieve relevant cards
cards = retrieval.search(
    query=f"{question}\nDIALECT=mysql",
    top_k=6,
    alpha=0.55  # 55% vector, 45% keyword
)

# 5. Build prompts
system_prompt = prompt_builder.build_system_prompt("mysql")
user_prompt = prompt_builder.build_user_prompt(question, cards, "mysql")

# 6. Generate SQL
raw_response = llm.chat(
    model="mistral:7b-instruct",
    system=system_prompt,
    user=user_prompt
)

# 7. Parse response
sections = parser.split_structured_output(raw_response)
sql = parser.strip_code_fences(sections["sql"])
explanation = sections["explanation"]
checks = parser.strip_code_fences(sections["checks"])

# 8. Validate
issues = validator.validate(sql)

# 9. Display results
print("=== Generated SQL ===")
print(sql)
print("\n=== Explanation ===")
print(explanation)
print("\n=== Validation ===")
if issues:
    for issue in issues:
        print(f"⚠️  {issue}")
else:
    print("✅ No issues found")
```

### Using Different LLM Providers

```python
from dbguide.services.llm_providers import OllamaProvider, OpenAIProvider

# Using Ollama (local)
ollama = OllamaProvider(base_url="http://localhost:11434")
response = ollama.chat(
    model="mistral:7b-instruct",
    system="You are a SQL expert.",
    user="How do I perform an INNER JOIN in MySQL?",
    temperature=0.2
)

# Using OpenAI (cloud)
openai = OpenAIProvider()  # Uses OPENAI_API_KEY from environment
response = openai.chat(
    model="gpt-4o-mini",
    system="You are a SQL expert.",
    user="How do I perform an INNER JOIN in MySQL?",
    temperature=0.2
)
```

### Hybrid Search with Metadata Filtering

```python
from dbguide.services.retrieval_service import HybridRetrievalService
from dbguide.services.indexing import BM25IndexBuilder
import chromadb

# Load indexes
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_collection(name="sql_cards")
bm25_index, documents = BM25IndexBuilder.load_index("data/bm25.pkl")

# Create retrieval service
retrieval = HybridRetrievalService(collection, bm25_index, documents)

# Search with metadata filter
results = retrieval.search(
    query="How to join tables in MySQL?",
    top_k=5,
    alpha=0.55,  # 55% vector, 45% keyword
    metadata_filter={"dialect": "mysql", "domain": "ecommerce"}
)

for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"ID: {result.id}")
    print(f"Text: {result.text[:200]}...")
    print("---")
```

### Adding Custom LLM Providers

You can easily extend the system with new LLM providers:

```python
from dbguide.domain.interfaces import LLMProvider
import anthropic

class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider implementation."""

    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(api_key=api_key)

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7
    ) -> str:
        response = self.client.messages.create(
            model=model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature
        )
        return response.content[0].text

# Use it
claude = AnthropicProvider()
response = claude.chat(
    model="claude-3-sonnet-20240229",
    system="You are a SQL expert.",
    user="How to optimize MySQL queries?"
)
```
### Contributing

- Follow SOLID principles (see [REFACTORING.md](REFACTORING.md))
- Use type annotations and docstrings
- Write tests for new features
- Update documentation
- Pull requests are welcome!

---

## 1. Installation

### Prerequisites
- Python 3.10+
- [UV](https://github.com/astral-sh/uv) package manager

### Install UV

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

In the project root:
```bash
uv sync
```

## 2. Configuration (.env)

Create a `.env` file in the root directory:

```bash
# Ollama (local)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL_MYSQL=mistral:7b-instruct
OLLAMA_MODEL_REDSHIFT=mistral:7b-instruct

# OpenAI (optional)
OPENAI_API_KEY=your_key_here
OPENAI_MODEL_MYSQL=gpt-4o-mini
OPENAI_MODEL_REDSHIFT=gpt-4o-mini
```

## 3. Build RAG Index

The cards are `.md` files in `dbguide/corpus/` (e.g., `pattern_cards/` and `query_cards/`).

To index the cards in ChromaDB + BM25:

```bash
uv run python build_indexes.py
```

This generates:
- `data/chroma/` - Vector index
- `data/bm25.pkl` - BM25 keyword index

## 4. Run the Application

### Option 1: Refactored App (Recommended)

```bash
uv run python run_app.py
```

```bash
uv run python run_app.py
```

Or directly:
The interface allows you to configure:

- **Dialect:** MySQL or Redshift (defines SQL generation context)
- **LLM Provider:** Ollama (local) or OpenAI (cloud)
- **Metadata Filter:**
  - *Heuristic*: Keyword-based filtering
  - *LLM*: AI-powered intelligent filtering
- **Cards (top_k):** Number of context cards to use
- **Vector vs Keyword Weight (alpha):** Balance between semantic and keyword search

Ask questions in natural language and get:
- Suggested SQL query
- Explanation in bullet points
- Validation checks for duplicates/volume
- Safety guardrails

## 5. How RAG Works (Summary)

1. `.md` files in `dbguide/corpus/` are indexed in:
   - ChromaDB (vector, via Sentence Transformers)
   - BM25 (keyword)

2. For each question, the app:
   - Retrieves most relevant cards (hybrid: vector + BM25)
   - Builds a prompt with safety rules + cards
   - Calls the model (Ollama or OpenAI) to generate SQL

## 6. LoRA Fine-tuning (Optional)

You can fine-tune Mistral 7B with LoRA for better SQL generation tailored to your data.

### Training Your Own Model

The project includes a professional LoRA training script with:
- ✅ 8-bit quantization for efficient training
- ✅ Comprehensive logging and validation
- ✅ Configurable hyperparameters
- ✅ Error handling and progress tracking

**Requirements:**
- GPU with 16GB+ VRAM (24GB recommended)
- Training dataset in JSONL format

**Dataset Format:**

Create a file `data/lora_training/train.jsonl` with examples:
```jsonl
{"instruction": "Show all users from last month", "response": "SELECT * FROM users WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 MONTH);"}
{"instruction": "Count orders by customer", "response": "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id;"}
```

A sample dataset with 25 examples is included in [data/lora_training/train.jsonl](data/lora_training/train.jsonl).

**Training:**

```bash
# Basic training (uses defaults)
uv run python scripts/lora_mistral7b_example.py

# Custom configuration
uv run python scripts/lora_mistral7b_example.py \
    --data-path data/my_training_data.jsonl \
    --output-dir data/my_lora_adapter \
    --epochs 3 \
    --lr 1e-4 \
    --batch-size 2 \
    --lora-r 16 \
    --lora-alpha 32

# See all options
uv run python scripts/lora_mistral7b_example.py --help
```

**Key Parameters:**
- `--epochs`: Number of training epochs (default: 1)
- `--lr`: Learning rate (default: 2e-4)
- `--batch-size`: Batch size per GPU (default: 1)
- `--grad-accum`: Gradient accumulation steps (default: 4)
- `--lora-r`: LoRA rank - higher = more parameters (default: 8)
- `--lora-alpha`: LoRA scaling (default: 16)
- `--max-length`: Maximum sequence length (default: 1024)

### Using Your Fine-tuned Model

After training, deploy the adapter to your inference server:

**Option 1: Ollama (Recommended)**

Create a Modelfile:
```dockerfile
FROM mistral:7b-instruct
ADAPTER ./data/lora_training/out_mistral7b_lora
```

Build and run:
```bash
ollama create mistral-sql -f Modelfile
```

Update `.env`:
```bash
OLLAMA_MODEL_MYSQL=mistral-sql
OLLAMA_MODEL_REDSHIFT=mistral-sql
```

**Option 2: vLLM**

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
    --lora-modules sql-adapter=./data/lora_training/out_mistral7b_lora
```

**Option 3: OpenAI Fine-tuned Models**

If using OpenAI's fine-tuning service:
```bash
OPENAI_MODEL_MYSQL=ft:gpt-4o-mini:org:model:id
OPENAI_MODEL_REDSHIFT=ft:gpt-4o-mini:org:model:id
```

### Training Tips

1. **Start small**: Test with 50-100 examples first
2. **Balance examples**: Include diverse query types (joins, aggregations, filters, etc.)
3. **Monitor metrics**: Watch loss curves - should decrease steadily
4. **Adjust learning rate**: If loss doesn't decrease, try lower LR (1e-4)
5. **Increase epochs**: 3-5 epochs often work better than 1
6. **Quality over quantity**: 100 high-quality examples > 1000 mediocre ones

### Troubleshooting

**Out of Memory (OOM)?**
- Reduce `--batch-size` to 1
- Increase `--grad-accum` to 8 or 16
- Reduce `--max-length` to 512
- Use `--no-8bit` for CPU training (very slow)

**Poor results?**
- Check dataset quality - ensure responses are correct SQL
- Increase `--epochs` to 3-5
- Add more diverse examples
- Adjust `--lora-r` to 16 or 32 (more capacity)

See [scripts/lora_mistral7b_example.py](scripts/lora_mistral7b_example.py) for implementation details.

---

## Testing

### Writing Unit Tests

```python
import unittest
from unittest.mock import Mock, patch
from dbguide.services.llm_providers import OllamaProvider

class TestOllamaProvider(unittest.TestCase):
    def setUp(self):
        self.provider = OllamaProvider(base_url="http://localhost:11434")

    @patch('requests.post')
    def test_chat_success(self, mock_post):
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "SELECT * FROM users;"}
        }
        mock_post.return_value = mock_response

        # Test
        result = self.provider.chat(
            model="mistral:7b-instruct",
            system="You are a SQL expert.",
            user="Show all users"
        )

        # Assert
        self.assertEqual(result, "SELECT * FROM users;")
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_chat_failure(self, mock_post):
        # Mock failed response
        mock_post.side_effect = Exception("Connection error")

        # Test
        with self.assertRaises(Exception):
            self.provider.chat(
                model="mistral:7b-instruct",
                system="You are a SQL expert.",
                user="Show all users"
            )

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
import chromadb
from dbguide.services.indexing import VectorIndexBuilder, BM25IndexBuilder
from dbguide.services.document_loader import DocumentLoader
from dbguide.services.retrieval_service import HybridRetrievalService

# Load documents
loader = DocumentLoader()
documents = loader.load_documents("dbguide/corpus/**/*.md")

# Build indexes
client = chromadb.Client()
collection = client.create_collection(name="test_cards")
vector_builder = VectorIndexBuilder()
vector_builder.build_index(collection, documents)

bm25_builder = BM25IndexBuilder()
bm25_index = bm25_builder.build_index(documents)

# Test retrieval
retrieval = HybridRetrievalService(collection, bm25_index, documents)
results = retrieval.search("How to join tables?", top_k=3)

assert len(results) > 0
assert results[0].score > 0
print(f"✅ Found {len(results)} results")
```

---

## Contributing

Contributions are welcome! To contribute:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Follow SOLID principles** when adding new features
4. **Use type annotations** and docstrings for all functions/classes
5. **Write tests** for new functionality
6. **Commit your changes:** `git commit -m 'Add amazing feature'`
7. **Push to the branch:** `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Code Style

- Use type hints for all function parameters and return values
- Write comprehensive docstrings (Google style)
- Follow PEP 8 conventions
- Keep functions focused (Single Responsibility Principle)
- Use dependency injection instead of hardcoded dependencies

### Architecture Guidelines

When extending the system:

1. **Add new interfaces in `domain/interfaces.py`** if you need a new abstraction
2. **Implement services in `services/`** following the interface
3. **Use factory functions** to create instances with proper dependencies
4. **Maintain backward compatibility** when possible
5. **Update tests** and documentation

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

