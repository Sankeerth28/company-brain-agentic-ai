# app_streamlit.py
"""
Streamlit demo UI for "Company Brain" (Agentic AI hackathon MVP).
Works on Windows. Assumes these project files exist in the same folder:
 - ingestion.py   (provides ingest_data())
 - rag_agent.py   (provides answer_and_propose(question) and retrieve())
 - data/          (optional; uploader will write files here)
 - db/            (created by ingestion.py)
.env may contain OPENAI_API_KEY and USE_MOCK_DEMO flags, but rag_agent.py defaults to mock mode.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

import streamlit as st

# Import local project functions
# ingestion.py must define ingest_data(data_dir="data", out_dir="db", model_name="all-MiniLM-L6-v2")
# rag_agent.py must define answer_and_propose(question, k=4) which returns dict with answer, actions, retrieved_context
from ingestion import ingest_data
from rag_agent import answer_and_propose, retrieve

# --- UI config ---
st.set_page_config(page_title="Company Brain — Agentic AI Demo", layout="wide")
st.title("Company Brain — Agentic AI")
st.markdown(
    "Agentic RAG demo: upload company docs, index them locally with sentence-transformers, "
    "ask questions, and get grounded answers + concrete next actions."
)

# Sidebar controls
st.sidebar.header("Controls / Setup")
with st.sidebar.form("setup"):
    model_name = st.text_input("Embedding model (local)", value="all-MiniLM-L6-v2")
    db_dir = st.text_input("Local DB folder", value="db")
    data_dir = st.text_input("Documents folder", value="data")
    use_mock_label = "Use mock responses (no OpenAI calls)"
    use_mock = st.checkbox(use_mock_label, value=True, help="Mock returns deterministic canned answers for demo.")
    run_setup = st.form_submit_button("Apply (note: changes require re-run if rag_agent uses env flags)")

if run_setup:
    # Note: rag_agent.py reads env at import time; to toggle real/mock reliably edit .env or restart app.
    st.success("Settings applied locally for this session. If you changed USE_MOCK_DEMO in .env, restart the app.")

st.markdown("---")

# left column: ingestion / upload
left, right = st.columns([1, 2])

with left:
    st.subheader("Ingest documents")
    st.markdown(
        "Upload PDFs / .txt / .md / .csv. Files are saved into the `data/` folder and then indexed locally "
        "with sentence-transformers. This uses CPU by default."
    )
    uploaded = st.file_uploader("Upload files (multiple)", accept_multiple_files=True, key="uploader")

    if st.button("Save uploaded files to data/"):
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        count = 0
        for f in uploaded or []:
            target = Path(data_dir) / f.name
            with open(target, "wb") as out:
                out.write(f.getbuffer())
            count += 1
        st.success(f"Saved {count} file(s) into `{data_dir}`. Now run indexing below.")

    st.markdown("**Index (build embeddings)**")
    if st.button("Run local indexing (ingest) — may take ~30s"):
        st.info("Indexing — this computes embeddings locally. Wait for completion.")
        with st.spinner("Indexing documents..."):
            try:
                # call the ingest_data function; passes data_dir and db_dir
                ingest_data(data_dir=data_dir, out_dir=db_dir, model_name=model_name)
                st.success(f"Ingestion complete — vector DB persisted to `{db_dir}`.")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.markdown("Files currently in data/ (for quick check):")
    try:
        files = sorted([p.name for p in Path(data_dir).iterdir() if p.is_file()])
        if files:
            for fn in files:
                st.text(f"- {fn}")
        else:
            st.info("No files found in data/. Use the uploader or add sample docs.")
    except Exception:
        st.info("No data/ folder found yet. Upload files or run ingestion after adding docs.")

    st.markdown("---")
    st.subheader("Helpful actions")
    if st.button("Remove local DB (force rebuild)"):
        if Path(db_dir).exists():
            try:
                # careful delete
                import shutil
                shutil.rmtree(db_dir)
                st.success(f"Removed `{db_dir}`. Re-run indexing to rebuild.")
            except Exception as e:
                st.error(f"Could not remove {db_dir}: {e}")
        else:
            st.info("No DB folder present.")

with right:
    st.subheader("Ask the Company Brain")
    question = st.text_input("Ask a question (examples below)", value="What are the top three customer complaints in Q3 and what should we do?")
    example_buttons = st.columns(3)
    if example_buttons[0].button("Top complaints (Q3)"):
        question = "What are the top three customer complaints in Q3 and what should we do?"
    if example_buttons[1].button("Sprint blockers -> actions"):
        question = "Which sprint blockers caused the biggest delays and what should engineering do?"
    if example_buttons[2].button("Churn reduction steps"):
        question = "Summarize churn reasons and propose 3 actionable steps to reduce churn (owner + est hours)."

    if st.button("Run Query"):
        # verify DB exists
        if not Path(db_dir).exists() or not Path(db_dir).joinpath("embeddings.npy").exists():
            st.warning("No local DB found. Run ingestion (indexing) first.")
        else:
            # Run answer_and_propose; show spinner
            with st.spinner("Retrieving context and reasoning..."):
                try:
                    res = answer_and_propose(question, k=4)
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    res = None

            if res:
                st.markdown("### Answer (concise)")
                st.write(res.get("answer", "No answer produced"))

                st.markdown("### Proposed concrete actions")
                st.write(res.get("actions", "No actions produced"))

                st.markdown("### Retrieved context (grounding)")
                # show retrieved_context in a readable collapsible
                rc = res.get("retrieved_context", "")
                if rc:
                    st.text(rc)
                else:
                    st.info("No retrieved context available from the retrieval step.")

    st.markdown("---")
    st.subheader("Quick manual retrieval (inspect top snippets)")
    inspect_q = st.text_input("Inspect retrieval for this query", value="", key="inspect_q")
    if st.button("Retrieve top snippets"):
        if not inspect_q:
            st.info("Type a short query to retrieve top snippets.")
        elif not Path(db_dir).exists():
            st.warning("No local DB found. Run ingestion first.")
        else:
            try:
                snippets = retrieve(inspect_q, k=6)
                st.write("Top retrieved snippets (source + score):")
                for i, s in enumerate(snippets):
                    src = s.get("meta", {}).get("source", "unknown")
                    score = s.get("score", 0)
                    st.markdown(f"**{i+1}.** Source: `{src}` — score: {score:.4f}")
                    st.write(s.get("text")[:1000])
                    st.markdown("---")
            except Exception as e:
                st.error(f"Retrieval failed: {e}")

st.markdown("---")
st.caption(
    "Notes: This demo uses local embeddings (sentence-transformers). "
    "By default rag_agent is in mock mode to avoid using OpenAI quota. "
    "To use real OpenAI calls, set USE_MOCK_DEMO=false and OPENAI_API_KEY in .env and restart the app."
)
