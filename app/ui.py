"""Streamlit chat UI. Loads an existing Chroma collection; does not re-ingest."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.config import settings
from app.rag import build_qa_chain, chroma_is_ready

st.set_page_config(page_title="LLMChat")
st.title("LLMChat")

if not settings.openai_api_key:
    st.error("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    st.stop()

if not chroma_is_ready():
    st.warning(
        "Vector store is empty. Run ingest first:\n\n"
        "`docker compose --profile ingest up ingest`"
    )
    st.stop()


@st.cache_resource
def get_qa_chain():
    return build_qa_chain()


qa = get_qa_chain()

user_input = st.text_input("Enter a query: ")
if user_input:
    query = f"###Prompt {user_input}"
    try:
        llm_response = qa(query)
        st.write(llm_response["result"])
    except Exception as err:
        st.error("Exception occurred. Please try again")
        print("Exception occurred. Please try again", str(err))
