Company Brain — Agentic AI (Hackathon MVP)
==========================================

**What this is:** A 48-hour hackathon prototype that demonstrates an _agentic_ AI: local RAG retrieval + reasoning + concrete next actions. Uses local sentence-transformer embeddings for retrieval and a simple query layer. rag\_agent.py supports mock (offline) mode by default to avoid API quota issues.

Project structure
-----------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   prosper-hack/  ├─ app_streamlit.py        # Streamlit demo UI  ├─ ingestion.py            # Ingest files -> chunk -> compute local embeddings -> persist in db/  ├─ rag_agent.py            # Retriever + (mock or live OpenAI) answer generator  ├─ data/                   # Put demo docs here (txt, md, csv, pdf)  ├─ db/                     # Persisted embeddings/texts after running ingestion.py  ├─ .env                    # Optional (OPENAI_API_KEY, USE_MOCK_DEMO=false to enable live LLM)  ├─ requirements.txt  └─ README.md   `

Quick start (Windows PowerShell)
--------------------------------

1.  Open **PowerShell** in project root (e.g. D:\\prosper-hack).
    
2.  python -m venv venv.\\venv\\Scripts\\Activate.ps1# If PowerShell blocks scripts, run once:# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
    
3.  pip install --upgrade pippip install -r requirements.txt
    
4.  (Optional) Put demo docs into data/:
    
    *   Drag-and-drop files into D:\\prosper-hack\\data or use the Streamlit uploader later.
        
5.  \# Remove old DB (safe)Remove-Item -Recurse -Force .\\db -ErrorAction SilentlyContinue# Run ingestion (creates db/embeddings.npy, texts.json, metadatas.json)python ingestion.pyYou should see output like:Found X files in data. Total chunks: N. Computing embeddings with all-MiniLM-L6-v2 (local)...Ingested N chunks. Files saved in db
    
6.  streamlit run app\_streamlit.pyOpen the URL printed (usually [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)). Upload / index files via the UI or use the files already in data/. Then ask queries.
    

Toggle mock vs live LLM
-----------------------

*   **Default:** rag\_agent.py uses **mock mode** (USE\_MOCK\_DEMO=true) — safe and deterministic for demos.
    
*   To enable live **OpenAI** calls:
    
    1.  USE\_MOCK\_DEMO=falseOPENAI\_API\_KEY=sk-...OPENAI\_MODEL=gpt-3.5-turbo
        
    2.  **Restart** Streamlit (or restart the app environment) so the env is reloaded.
        

**Note:** If you run live calls, keep them minimal during the demo to avoid quota/latency risk.

Re-run / Debug checklist
------------------------

*   Remove-Item -Recurse -Force .\\dbpython ingestion.py
    
*   streamlit run app\_streamlit.py
    
*   pip install -r requirements.txt
    
*   If a model download stalls during ingestion, give it a minute — the sentence-transformers model downloads once.
    

Common issues & fixes
---------------------

*   **PowerShell execution blocked:** run Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force.
    
*   **API rate limit / quota (OpenAI):** use mock mode (default) or set USE\_MOCK\_DEMO=true.
    
*   **Missing db/ or embeddings.npy:** run python ingestion.py to rebuild.
    
*   **Slow first run:** sentence-transformers downloads the model on first embed; subsequent runs are fast.
    

How to demo (60–90s)
--------------------

1.  Open app. Show “Ingest documents” panel and mention local indexing.
    
2.  Ask: “What are the top three customer complaints in Q3 and what should we do?” -> Click Run Query.
    
3.  Point at **Answer** (concise), **Proposed actions** (owner + est hours), and **Retrieved context** (grounding).
    
4.  Closing line: **“It doesn’t just answer — it acts: planning, prioritizing, and proposing concrete next steps.”**
    

Notes for judges / tech questions
---------------------------------

*   **Retrieval:** local sentence-transformer embeddings + cosine similarity.
    
*   **Reasoning:** either mock canned actions or live LLM (OpenAI) that consumes retrieved context.
    
*   **Scale path:** swap in a cloud vector DB (Chroma/Weaviate/FAISS), scale embeddings generation to batch workers, or plug in hosted LLMs for production reasoning.