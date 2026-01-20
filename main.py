import os
import pdfplumber
import re
import nltk # Sentence-level tokenization
import faiss # Vector Database
import numpy as np
from sentence_transformers import SentenceTransformer # Embeddings
from openai import OpenAI
from nltk.tokenize import sent_tokenize

# ---------------- CONFIGURATION ----------------
DEBUG = True

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PDF_PATH = r"C:\Users\ASUS\PycharmProjects\EEE517_Project\A_Darker_Shade_of_Magic.pdf"

INDEX_FILE = "book_index.faiss" # stores the vector DB
CHUNKS_FILE = "chunks.npy" # text chunks

# ---------------- SETUP! ----------------
nltk.download("punkt")
nltk.download("punkt_tab")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

embedder = SentenceTransformer("all-MiniLM-L6-v2") # Produces 384-dimensional embeddings

# ---------------- CREATING VECTOR DB ----------------
def get_vector_db():

    #One-time preprocessing step:
    #- PDF loading
    #- Chunking
    #- Embedding
    #- FAISS indexing

    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("\n[INFO] Loading existing FAISS index from disk (one-time job already done).")
        index = faiss.read_index(INDEX_FILE)
        chunks = np.load(CHUNKS_FILE, allow_pickle=True).tolist()
        return index, chunks

    print("\n Loading PDF...")
    text = ""
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    print(f"[INFO] Raw text length: {len(text)} characters")
    clean_text = re.sub(r"\s+", " ", text).strip()

    print("\n Sentence tokenization...")
    sentences = sent_tokenize(clean_text) # sentence-level BERT/splitting
    print(f"[INFO] Total sentences extracted: {len(sentences)}")

    print("\n Chunking sentences (~600 chars per chunk)...")
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= 600:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())

    print(f"[INFO] Total chunks created: {len(chunks)}")

    if DEBUG:
        print("\n[SAMPLE CHUNKS]")
        for i, c in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i} (length={len(c)}) ---")
            print(c[:300], "...")

    print("\n Embedding chunks into dense vectors...")
    embeddings = embedder.encode(chunks, show_progress_bar=True) # Numpy array with the shape: (# of chunks, 384)

    print("[INFO] Embedding matrix shape:", embeddings.shape)
    print("[INFO] Each chunk →", embeddings.shape[1], "dimensional vector")

    if DEBUG:
        print("First 5 values of first embedding vector:")
        print(embeddings[0][:5]) #embeddings are being generated correctly and are non-zero

    print("\n Building FAISS vector index (cosine similarity)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) # initialize using inner product similarity
    faiss.normalize_L2(embeddings) # normalized to unit length!
    index.add(embeddings)

    print("[INFO] FAISS index contains", index.ntotal, "vectors")

    print("\n Saving index and chunks for reuse...")
    faiss.write_index(index, INDEX_FILE)
    np.save(CHUNKS_FILE, np.array(chunks, dtype=object))

    print("[INFO] Vector database ready (one-time preprocessing completed).")
    return index, chunks


# Initialize vector DB
index, chunks = get_vector_db()

# ---------------- RAG ----------------
def answer_with_rag(query):
    print("\n Encoding query and performing similarity search...")
    # query encoding
    q_emb = embedder.encode([query])
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k=5)

    if DEBUG:
        print("\n[TOP RETRIEVED CHUNKS]")
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            print(f"\nRank {rank+1}")
            print(f"Chunk ID: {idx}")
            print(f"Cosine similarity: {score:.4f}")
            print(chunks[idx][:250], "...")

    #Convert the passages into a single structured context string
    retrieved_context = "\n\n---\n\n".join([chunks[i] for i in indices[0]])

    print("\n External data passed to LLM:")
    if DEBUG:
        print(retrieved_context[:1200])
    else:
        print("(hidden, DEBUG=False)")

    prompt = f"""
To answer the Question, look at the External Data only.
If you cannot answer the Question, write «I do not know the answer.»

External Data:
{retrieved_context}

Question: {query}
"""

    print("\n Sending prompt to OpenRouter LLM...")
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] LLM request failed: {str(e)}"


# ---------------- MAIN LOOP ----------------
if __name__ == "__main__":
    print("\n===== RAG SYSTEM READY =====")
    print("Book: A Darker Shade of Magic")
    print("============================")

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue

        print("\n[QUERY RECEIVED]")
        answer = answer_with_rag(q)
        print("\n[FINAL ANSWER]")
        print(answer)
