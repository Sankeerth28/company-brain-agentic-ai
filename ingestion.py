# ingestion.py  -- LangChain-free ingestion using sentence-transformers + simple chunking
import os, json, csv
from pathlib import Path

# local embedding model
from sentence_transformers import SentenceTransformer
import numpy as np

# pdf reading
def load_pdf_as_text(path):
    try:
        from pypdf import PdfReader
    except Exception:
        from PyPDF2 import PdfReader
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

def load_file_docs(path):
    path = Path(path)
    ext = path.suffix.lower()
    docs = []
    if ext == ".pdf":
        text = load_pdf_as_text(str(path))
        docs.append({"text": text, "source": path.name})
    elif ext in (".txt", ".md"):
        text = path.read_text(encoding="utf-8")
        docs.append({"text": text, "source": path.name})
    elif ext == ".csv":
        with open(path, newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return []
            header = rows[0]
            for i, row in enumerate(rows[1:], start=1):
                text = ", ".join(f"{h}: {v}" for h, v in zip(header, row))
                docs.append({"text": text, "source": path.name, "row": i})
    else:
        try:
            text = path.read_text(encoding="utf-8")
            docs.append({"text": text, "source": path.name})
        except Exception:
            print(f"Skipping {path} (unsupported)")
    return docs

def chunk_text(text, max_chars=800, overlap=100):
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks

def ingest_data(data_dir="data", out_dir="db", model_name="all-MiniLM-L6-v2"):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([p for p in Path(data_dir).iterdir() if p.is_file()])
    all_texts = []
    metadatas = []
    print(f"Found {len(files)} files in {data_dir}. Extracting...")
    for p in files:
        docs = load_file_docs(p)
        for doc in docs:
            chunks = chunk_text(doc["text"])
            for i, c in enumerate(chunks):
                all_texts.append(c)
                meta = {"source": doc.get("source"), "chunk_index": i}
                if "row" in doc:
                    meta["row"] = doc["row"]
                metadatas.append(meta)

    print(f"Total chunks: {len(all_texts)}. Computing embeddings with {model_name} (local)...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True)

    # persist
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    # texts and metadata as json
    with open(os.path.join(out_dir, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(all_texts, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "metadatas.json"), "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False)

    print(f"Ingested {len(all_texts)} chunks. Files saved in {out_dir}")

if __name__ == "__main__":
    ingest_data()
