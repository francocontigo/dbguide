from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import chromadb
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Ensure the internal "dbguide" package is visible as a top-level package
# by adding the project root directory to sys.path.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]  # .../dev/dbguide
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dbguide.app.llm import ollama_chat, openai_chat
from dbguide.app.prompts import system_prompt, user_prompt
from dbguide.app.retrieval import load_bm25, hybrid_search
from dbguide.app.safety import (
    basic_sql_safety_check,
    extract_sql_block,
    split_structured_output,
)

load_dotenv()

# Must be called before any other Streamlit command
st.set_page_config(page_title="DBGuide", layout="wide")

MODEL_MYSQL_OLLAMA = os.getenv("OLLAMA_MODEL_MYSQL", "mistral:7b-instruct")
MODEL_REDSHIFT_OLLAMA = os.getenv("OLLAMA_MODEL_REDSHIFT", "mistral:7b-instruct")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

MODEL_MYSQL_OPENAI = os.getenv("OPENAI_MODEL_MYSQL", "gpt-4o-mini")
MODEL_REDSHIFT_OPENAI = os.getenv("OPENAI_MODEL_REDSHIFT", "gpt-4o-mini")


@st.cache_resource
def load_retrieval():
    client = chromadb.PersistentClient(path="data/chroma")
    col = client.get_or_create_collection(name="sql_cards")
    bm25, docs = load_bm25("data/bm25.pkl")
    return col, bm25, docs


def pick_model(dialect: str, provider: str) -> str:
    if provider == "OpenAI":
        return MODEL_MYSQL_OPENAI if dialect == "mysql" else MODEL_REDSHIFT_OPENAI
    return MODEL_MYSQL_OLLAMA if dialect == "mysql" else MODEL_REDSHIFT_OLLAMA


# --- HIDE STREAMLIT UI ---
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("🧠 DBGuide")
st.caption("Chat para gerar SQL seguro usando RAG.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_cards" not in st.session_state:
    st.session_state["last_cards"] = []
if "last_sql" not in st.session_state:
    st.session_state["last_sql"] = ""
if "last_guardrails" not in st.session_state:
    st.session_state["last_guardrails"] = ""
if "last_raw" not in st.session_state:
    st.session_state["last_raw"] = ""
if "last_explicacao" not in st.session_state:
    st.session_state["last_explicacao"] = ""
if "last_checks" not in st.session_state:
    st.session_state["last_checks"] = ""

if "provider_state" not in st.session_state:
    # Store history and last result for each provider (Ollama / OpenAI)
    st.session_state["provider_state"] = {}


def _strip_code_fences(text: str | None) -> str:
    """Remove all ```...``` fences (with or without language labels).

    Works even when there are multiple code blocks in the same text.
    """
    if not text:
        return ""

    lines = text.strip().splitlines()
    clean_lines = [ln for ln in lines if not ln.strip().startswith("```")]
    return "\n".join(clean_lines).strip()


def _get_provider_state(provider: str) -> dict:
    """Return (or initialize) the state associated with a given LLM provider."""
    ps = st.session_state.setdefault("provider_state", {})
    if provider not in ps:
        ps[provider] = {
            "messages": [],
            "last_cards": [],
            "last_sql": "",
            "last_guardrails": "",
            "last_raw": "",
            "last_explicacao": "",
            "last_checks": "",
        }
    return ps[provider]

with st.sidebar:
    st.header("Configuracoes")
    dialect = st.selectbox("Dialeto", ["mysql", "redshift"], index=0)
    provider = st.radio("Provedor de LLM", ["Ollama", "OpenAI"], index=0)

    st.markdown("---")
    st.subheader("Retrieval (RAG)")
    top_k = st.slider("Cards (top_k)", min_value=3, max_value=12, value=6, step=1)
    alpha = st.slider("Peso vetor vs keyword (alpha)", min_value=0.0, max_value=1.0, value=0.55, step=0.05)

    if provider == "OpenAI" and not os.getenv("OPENAI_API_KEY"):
        st.warning("Set OPENAI_API_KEY in .env to use OpenAI.")

# Sincroniza o estado global com o estado especifico do provedor selecionado
_prov_state = _get_provider_state(provider)
st.session_state["messages"] = _prov_state["messages"]
st.session_state["last_cards"] = _prov_state["last_cards"]
st.session_state["last_sql"] = _prov_state["last_sql"]
st.session_state["last_guardrails"] = _prov_state["last_guardrails"]
st.session_state["last_raw"] = _prov_state["last_raw"]
st.session_state["last_explicacao"] = _prov_state["last_explicacao"]
st.session_state["last_checks"] = _prov_state["last_checks"]


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def handle_question(question: str) -> None:
    if not question.strip():
        return

    # Provider-specific state for the currently selected provider
    prov_state = _get_provider_state(provider)

    # For each new question, reset the conversation history for this provider
    prov_state["messages"] = []
    prov_state["messages"].append({"role": "user", "content": question})
    st.session_state["messages"] = prov_state["messages"]

    with st.spinner("☝️🤓 Gerando SQL ..."):
        col, bm25, docs = load_retrieval()

        retrieval_query = f"{question}\nDIALETO={dialect}"
        cards = hybrid_search(retrieval_query, col, bm25, docs, top_k=top_k, alpha=alpha)
        prov_state["last_cards"] = cards

        sys = system_prompt(dialect)
        usr = user_prompt(question, cards, dialect)

        model = pick_model(dialect, provider)

        if provider == "OpenAI":
            raw = openai_chat(model=model, system=sys, user=usr)
        else:
            raw = ollama_chat(model=model, system=sys, user=usr, ollama_url=OLLAMA_URL)

        sections = split_structured_output(raw)
        sql_only = sections.get("sql") or extract_sql_block(raw)
        explicacao_text = sections.get("explicacao", "")
        checks_text = sections.get("checks", "")

        # Ensure SQL and checks do not contain ```sql ... ``` fences inside the block.
        sql_only = _strip_code_fences(sql_only)
        checks_text = _strip_code_fences(checks_text)

        issues = basic_sql_safety_check(sql_only)

    if issues:
        guardrails_text = "\n".join(f"- {i}" for i in issues)
    else:
        guardrails_text = "- Sem violacoes obvias (MVP)."

    prov_state["last_sql"] = sql_only
    prov_state["last_guardrails"] = guardrails_text
    prov_state["last_raw"] = raw
    prov_state["last_explicacao"] = explicacao_text
    prov_state["last_checks"] = checks_text

    # Reflete tambem no estado global (para rendering imediato)
    # Mirror provider-specific state into the global session state for immediate rendering
    st.session_state["last_cards"] = prov_state["last_cards"]
    st.session_state["last_sql"] = prov_state["last_sql"]
    st.session_state["last_guardrails"] = prov_state["last_guardrails"]
    st.session_state["last_raw"] = prov_state["last_raw"]
    st.session_state["last_explicacao"] = prov_state["last_explicacao"]
    st.session_state["last_checks"] = prov_state["last_checks"]


question = st.chat_input("Descreva seu pedido em linguagem natural...")
if question is not None:
    handle_question(question)


if st.session_state["last_sql"]:
    st.markdown("---")
    st.subheader("SQL sugerido e resultado")

    sql_text = st.session_state["last_sql"]
    guardrails_text = st.session_state.get("last_guardrails", "")
    explicacao_text = st.session_state.get("last_explicacao", "")
    checks_text = st.session_state.get("last_checks", "")

    # Usa apenas o bloco de codigo do Streamlit (que ja tem icone de copiar)
    st.code(sql_text, language="sql")

    st.markdown("**Guardrails (checks de segurança e qualidade)**")
    st.markdown(guardrails_text or "- Sem violacoes obvias (MVP).")

    if explicacao_text.strip():
        st.markdown("---")
        st.markdown("**Explicação**")
        st.markdown(explicacao_text)

    if checks_text.strip():
        st.markdown("**Checks (SQL para validação)**")
        st.code(checks_text, language="sql")


if st.session_state["last_cards"]:
    st.markdown("---")
    st.subheader("Cards usados (RAG)")
    for c in st.session_state["last_cards"]:
        with st.expander(f"{os.path.basename(c['id'])} - score={c['score']:.2f}"):
            st.markdown(c["text"][:8000])
