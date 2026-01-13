"""
DBGuide Streamlit Application.
Refactored with dependency injection and SOLID principles.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import streamlit as st
from dotenv import load_dotenv

from dbguide.models.document import Document
from dbguide.services.document_loader import get_all_metadata_keys_and_values
from dbguide.services.indexing import BM25IndexBuilder
from dbguide.services.llm_providers import OllamaProvider, OpenAIProvider
from dbguide.services.metadata_filter import create_metadata_filter
from dbguide.services.prompt_builder import SQLPromptBuilder
from dbguide.services.retrieval_service import HybridRetrievalService
from dbguide.services.sql_validator import BasicSQLValidator, SQLOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("dbguide")

load_dotenv()

# Page configuration
st.set_page_config(page_title="DBGuide", layout="wide")

# Environment variables
MODEL_MYSQL_OLLAMA = os.getenv("OLLAMA_MODEL_MYSQL", "mistral:7b-instruct")
MODEL_REDSHIFT_OLLAMA = os.getenv("OLLAMA_MODEL_REDSHIFT", "mistral:7b-instruct")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

MODEL_MYSQL_OPENAI = os.getenv("OPENAI_MODEL_MYSQL", "gpt-4o-mini")
MODEL_REDSHIFT_OPENAI = os.getenv("OPENAI_MODEL_REDSHIFT", "gpt-4o-mini")


class AppState:
    """Manages application state following the Single Responsibility Principle."""

    def __init__(self):
        """Initialize session state keys if they don't exist."""
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        if "last_cards" not in st.session_state:
            st.session_state["last_cards"] = []

        if "last_sql" not in st.session_state:
            st.session_state["last_sql"] = ""

        if "last_guardrails" not in st.session_state:
            st.session_state["last_guardrails"] = ""

        if "last_explanation" not in st.session_state:
            st.session_state["last_explanation"] = ""

        if "last_checks" not in st.session_state:
            st.session_state["last_checks"] = ""

        if "provider_state" not in st.session_state:
            st.session_state["provider_state"] = {}

    def get_provider_state(self, provider: str) -> Dict:
        """
        Get or create state for a specific provider.

        Args:
            provider: Provider name ('Ollama' or 'OpenAI').

        Returns:
            Provider-specific state dictionary.
        """
        ps = st.session_state["provider_state"]
        if provider not in ps:
            ps[provider] = {
                "messages": [],
                "last_cards": [],
                "last_sql": "",
                "last_guardrails": "",
                "last_explanation": "",
                "last_checks": "",
            }
        return ps[provider]

    def sync_provider_state(self, provider: str) -> None:
        """
        Sync global state with provider-specific state.

        Args:
            provider: Provider name to sync.
        """
        prov_state = self.get_provider_state(provider)
        st.session_state["messages"] = prov_state["messages"]
        st.session_state["last_cards"] = prov_state["last_cards"]
        st.session_state["last_sql"] = prov_state["last_sql"]
        st.session_state["last_guardrails"] = prov_state["last_guardrails"]
        st.session_state["last_explanation"] = prov_state["last_explanation"]
        st.session_state["last_checks"] = prov_state["last_checks"]


@st.cache_resource
def load_retrieval_resources():
    """
    Load retrieval resources (ChromaDB, BM25, documents).
    Cached to avoid reloading on every interaction.

    Returns:
        Tuple of (ChromaDB collection, BM25 index, documents list).
    """
    client = chromadb.PersistentClient(path="data/chroma")
    collection = client.get_or_create_collection(name="sql_cards")

    bm25_index, documents = BM25IndexBuilder.load_index("data/bm25.pkl")

    return collection, bm25_index, documents


def pick_model(dialect: str, provider: str) -> str:
    """
    Pick the appropriate model based on dialect and provider.

    Args:
        dialect: SQL dialect ('mysql' or 'redshift').
        provider: LLM provider ('Ollama' or 'OpenAI').

    Returns:
        Model name string.
    """
    if provider == "OpenAI":
        return MODEL_MYSQL_OPENAI if dialect == "mysql" else MODEL_REDSHIFT_OPENAI
    return MODEL_MYSQL_OLLAMA if dialect == "mysql" else MODEL_REDSHIFT_OLLAMA


def render_sidebar() -> tuple[str, str, str, int, float]:
    """
    Render sidebar with configuration options.

    Returns:
        Tuple of (dialect, provider, filter_mode, top_k, alpha).
    """
    with st.sidebar:
        st.header("Settings")

        dialect = st.selectbox("Dialect", ["mysql", "redshift"], index=0)
        provider = st.selectbox("LLM Provider", ["Ollama", "OpenAI"], index=0)
        filter_mode = st.selectbox(
            "Metadata Filter:",
            ["Heuristic", "LLM"],
            index=0,
            help="Choose if metadata filter will be heuristic (keyword) or suggested by LLM."
        )

        st.markdown("---")
        st.subheader("Retrieval (RAG)")
        top_k = st.slider("Cards (top_k)", min_value=3, max_value=12, value=6, step=1)
        alpha = st.slider(
            "Vector vs keyword weight (alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05
        )

        if provider == "OpenAI" and not os.getenv("OPENAI_API_KEY"):
            st.warning("Set OPENAI_API_KEY in .env to use OpenAI.")

    return dialect, provider, filter_mode, top_k, alpha


def handle_question(
    question: str,
    dialect: str,
    provider: str,
    filter_mode: str,
    top_k: int,
    alpha: float,
    app_state: AppState,
) -> None:
    """
    Handle user question and generate SQL response.

    Args:
        question: User's question.
        dialect: SQL dialect.
        provider: LLM provider name.
        filter_mode: Metadata filter mode.
        top_k: Number of cards to retrieve.
        alpha: Weight for vector vs keyword search.
        app_state: Application state manager.
    """
    if not question.strip():
        return

    # Get provider-specific state
    prov_state = app_state.get_provider_state(provider)

    # Reset conversation history for new question
    prov_state["messages"] = []
    prov_state["messages"].append({"role": "user", "content": question})
    st.session_state["messages"] = prov_state["messages"]

    with st.spinner("☝️🤓 Generating SQL ..."):
        # Load retrieval resources
        collection, bm25_index, documents = load_retrieval_resources()

        # Create services with dependency injection
        retrieval_service = HybridRetrievalService(collection, bm25_index, documents)

        # Get available metadata
        all_metadata = get_all_metadata_keys_and_values(documents)
        logger.info(f"[Agent] Available metadata for filter: {all_metadata}")

        # Create LLM provider for metadata filtering if needed
        if filter_mode == "LLM":
            filter_llm = OpenAIProvider()
            metadata_filter_service = create_metadata_filter(
                "llm",
                llm_provider=filter_llm,
                model="gpt-3.5-turbo"
            )
        else:
            metadata_filter_service = create_metadata_filter("heuristic")

        # Suggest metadata filter
        metadata_filter = metadata_filter_service.suggest_filter(
            question,
            all_metadata
        )
        logger.info(f"[Agent] Suggested filter: {metadata_filter}")

        # Perform retrieval
        retrieval_query = f"{question}\nDIALECT={dialect}"
        cards = retrieval_service.search(
            retrieval_query,
            top_k=top_k,
            alpha=alpha,
            metadata_filter=metadata_filter
        )

        # Limit to 3 pattern cards and 3 query cards
        pattern_cards = [
            c for c in cards
            if "/pattern_cards/" in c.id or "\\pattern_cards\\" in c.id
        ]
        query_cards = [
            c for c in cards
            if "/query_cards/" in c.id or "\\query_cards\\" in c.id
        ]

        limited_cards = pattern_cards[:3] + query_cards[:3]

        # Fill remaining slots if needed
        if len(limited_cards) < top_k:
            others = [c for c in cards if c not in limited_cards]
            limited_cards += others[:(top_k - len(limited_cards))]

        prov_state["last_cards"] = limited_cards

        # Build prompts
        prompt_builder = SQLPromptBuilder()
        system_prompt = prompt_builder.build_system_prompt(dialect)
        user_prompt = prompt_builder.build_user_prompt(question, limited_cards, dialect)

        # Create LLM provider for SQL generation
        model = pick_model(dialect, provider)

        if provider == "OpenAI":
            llm_provider = OpenAIProvider()
        else:
            llm_provider = OllamaProvider(base_url=OLLAMA_URL)

        # Generate SQL
        raw_response = llm_provider.chat(
            model=model,
            system=system_prompt,
            user=user_prompt
        )

        # Parse response
        parser = SQLOutputParser()
        sections = parser.split_structured_output(raw_response)

        sql_only = sections.get("sql") or parser.extract_sql_block(raw_response)
        explanation_text = sections.get("explanation", "")
        checks_text = sections.get("checks", "")

        # Strip code fences
        sql_only = parser.strip_code_fences(sql_only)
        checks_text = parser.strip_code_fences(checks_text)

        # Validate SQL
        validator = BasicSQLValidator()
        issues = validator.validate(sql_only)

        if issues:
            guardrails_text = "\n".join(f"- {issue}" for issue in issues)
        else:
            guardrails_text = "- No obvious violations (MVP)."

        # Update provider state
        prov_state["last_sql"] = sql_only
        prov_state["last_guardrails"] = guardrails_text
        prov_state["last_explanation"] = explanation_text
        prov_state["last_checks"] = checks_text

        # Sync to global state
        st.session_state["last_cards"] = prov_state["last_cards"]
        st.session_state["last_sql"] = prov_state["last_sql"]
        st.session_state["last_guardrails"] = prov_state["last_guardrails"]
        st.session_state["last_explanation"] = prov_state["last_explanation"]
        st.session_state["last_checks"] = prov_state["last_checks"]


def render_results() -> None:
    """Render SQL results and retrieved cards."""
    if st.session_state["last_sql"]:
        st.markdown("---")
        st.subheader("Suggested SQL and Results")

        # Display SQL
        st.code(st.session_state["last_sql"], language="sql")

        # Display guardrails
        st.markdown("**Guardrails (safety and quality checks)**")
        st.markdown(st.session_state.get("last_guardrails", "- No obvious violations (MVP)."))

        # Display explanation
        explanation = st.session_state.get("last_explanation", "").strip()
        if explanation:
            st.markdown("---")
            st.markdown("**Explanation**")
            st.markdown(explanation)

        # Display checks
        checks = st.session_state.get("last_checks", "").strip()
        if checks:
            st.markdown("**Checks (SQL for validation)**")
            st.code(checks, language="sql")

    # Display retrieved cards
    if st.session_state["last_cards"]:
        st.markdown("---")
        st.subheader("Cards used (RAG)")
        for card in st.session_state["last_cards"]:
            card_id = os.path.basename(card.id) if hasattr(card, 'id') else card.get('id', 'unknown')
            card_score = card.score if hasattr(card, 'score') else card.get('score', 0)
            card_text = card.text if hasattr(card, 'text') else card.get('text', '')

            with st.expander(f"{card_id} - score={card_score:.2f}"):
                st.markdown(card_text[:8000])


def main() -> None:
    """Main application entry point."""
    # Hide Streamlit UI elements
    hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Title
    st.title("🧠 DBGuide")
    st.caption("Chat to generate safe SQL using RAG.")

    # Initialize app state
    app_state = AppState()

    # Render sidebar and get configuration
    dialect, provider, filter_mode, top_k, alpha = render_sidebar()

    # Sync provider state
    app_state.sync_provider_state(provider)

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle user input
    question = st.chat_input("Describe your request in natural language...")
    if question:
        handle_question(
            question,
            dialect,
            provider,
            filter_mode,
            top_k,
            alpha,
            app_state
        )

    # Render results
    render_results()


if __name__ == "__main__":
    main()
