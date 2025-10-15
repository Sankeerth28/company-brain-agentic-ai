# rag_agent.py -- simple cosine-retrieval + optional OpenAI answer generation / mock mode
import os, json, numpy as np
from dotenv import load_dotenv
load_dotenv()

USE_MOCK = os.getenv("USE_MOCK_DEMO", "true").lower() in ("1","true","yes")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "all-MiniLM-L6-v2")

# local sentence-transformer model (for query embedding)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(EMB_MODEL_NAME)

# load persisted store
def load_store(db_dir="db"):
    emb = np.load(os.path.join(db_dir, "embeddings.npy"))
    with open(os.path.join(db_dir, "texts.json"), "r", encoding="utf-8") as f:
        texts = json.load(f)
    with open(os.path.join(db_dir, "metadatas.json"), "r", encoding="utf-8") as f:
        metas = json.load(f)
    return emb, texts, metas

def cosine_sim(a, b):
    # a: (d,), b: (n, d) -> returns (n,)
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b_norm, a_norm)

def retrieve(question, k=4, db_dir="db"):
    emb, texts, metas = load_store(db_dir)
    qv = model.encode([question], convert_to_numpy=True)[0]
    sims = cosine_sim(qv, emb)
    topk_idx = np.argsort(-sims)[:k]
    results = [{"text": texts[i], "meta": metas[i], "score": float(sims[i])} for i in topk_idx]
    return results

# optional OpenAI call (if not using mock)
def answer_and_propose(question, k=4):
    results = retrieve(question, k=k)
    context = "\n\n".join(f"[{r['meta'].get('source','doc')}] {r['text'][:800]}" for r in results)

    if USE_MOCK:
        # a canned response for demo queries — adjust if you want different wording
        q = question.lower()
        if "top three customer complaints" in q or "top three customer complaints in q3" in q:
            answer = ("Top complaints in Q3: 1) CSV export crashes on reports with inline comments (release_notes_q3.md, customer_email_1.txt); "
                      "2) Confusing billing/tier changes (customer_email_2.txt, churn_survey_responses.csv); "
                      "3) API/timeouts for bulk uploads (support_tickets.csv).")
            actions = ("1) Request failing report + repro steps - Support - 4h - Get repro to unblock engineering.\n"
                       "2) Add billing tier explanation + rollback flow - Product - 8h - Reduce churn and urgent support load.\n"
                       "3) Triage API incidents & allocate hotfix engineer - Eng Lead - 12h - Prioritize enterprise-impact issues.")
        else:
            answer = "Demo answer (mock): couldn't match the question to a canned output."
            actions = "1) Investigate - Owner - 4h - Demo action."
        return {"answer": answer, "actions": actions, "retrieved_context": context}

    # live OpenAI path (if you have key and want to use it)
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    system = ("You are Company Brain — an autonomous analyst. Use the given context to answer succinctly (<=200 words), "
              "cite up to 2 sources by filename in brackets, then propose exactly 3 concrete next actions "
              "formatted as: 1) Action - Owner - Est_Hours - One-sentence rationale.")
    user = f"Context:\n{context}\n\nQuestion: {question}"
    resp = openai.ChatCompletion.create(model=model_name, messages=[{"role":"system","content":system}, {"role":"user","content":user}], temperature=0.0, max_tokens=600)
    answer = resp["choices"][0]["message"]["content"].strip()

    # followup actions pass
    followup_prompt = ("You are an autonomous analyst. Based on the short answer below, produce exactly 3 next actions.\n\n"
                       f"Answer:\n{answer}\n\nFormat:\n1) Action - Owner - Est_Hours - One-sentence rationale")
    resp2 = openai.ChatCompletion.create(model=model_name, messages=[{"role":"system","content":"You are pragmatic."},{"role":"user","content":followup_prompt}], temperature=0.0, max_tokens=300)
    actions = resp2["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "actions": actions, "retrieved_context": context}
