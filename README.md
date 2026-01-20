# RAG-Book-QA

**Retrieval-Augmented Generation for Book-Based Question Answering**

---

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that answers questions **strictly based on the content of a book**, rather than relying on a language model’s internal knowledge.

The system was developed as part of the **EEE517 course project**, with the goal of:

* Preventing hallucinations
* Ensuring explainable and grounded answers
* Applying modern NLP and deep learning concepts in practice

The chosen dataset is the novel ***A Darker Shade of Magic* by V. E. Schwab**, provided as a PDF file.

---

## Motivation

Large Language Models (LLMs) often generate answers that:

* Are outdated
* Are factually incorrect
* Hallucinate information not present in the source

To address this, the project follows a **Retrieval-Augmented Generation** approach:

* The model is **not allowed** to answer unless the information exists in the retrieved book passages.
* If relevant information is missing, the system explicitly replies:

> *“I do not know the answer.”*

---

## System Architecture (RAG Pipeline)

The system follows the standard RAG pipeline taught in the course:

```
PDF Book
  ↓
Text Extraction & Cleaning
  ↓
Sentence Tokenization
  ↓
Chunking (~600 characters)
  ↓
Sentence-BERT Embeddings
  ↓
FAISS Vector Database
  ↓
Query Embedding
  ↓
Cosine Similarity Search (Top-5)
  ↓
Context Injection into LLM
  ↓
Grounded Answer Generation
```

---

## Tools & Technologies Used

| Component            | Tool / Library                     |
| -------------------- | ---------------------------------- |
| PDF Processing       | `pdfplumber`                       |
| Text Cleaning        | `re` (Regular Expressions)         |
| Tokenization         | `nltk`                             |
| Embeddings           | `Sentence-BERT (all-MiniLM-L6-v2)` |
| Vector Database      | `FAISS`                            |
| Similarity Metric    | Cosine Similarity                  |
| LLM Backend          | OpenRouter (OpenAI-compatible API) |
| Programming Language | Python                             |

---

## Dataset Characteristics

* **Source:** PDF version of *A Darker Shade of Magic*
* **Type:** Unstructured narrative text
* **Size:** Full-length novel
* **Preprocessing:**

  * Whitespace normalization
  * Sentence-level tokenization
  * Chunking into ~600-character passages
* **Storage:**

  * Text chunks cached in `chunks.npy`
  * Vector index stored in `book_index.faiss`

---

## Implementation Details

### PDF Loading & Text Cleaning

* The book is read page-by-page using `pdfplumber`
* Non-textual elements are ignored
* Regular expressions normalize whitespace to ensure clean input for NLP processing

---

### Sentence Tokenization

* NLTK’s `sent_tokenize` is used to split text into sentences
* Sentence-level splitting preserves semantic meaning better than raw character slicing

---

### Chunking Strategy

Multiple chunk sizes were tested:

* **400 characters** → Too fragmented
* **800 characters** → Semantic dilution
* **600 characters** → Best balance

The final system uses **~600-character chunks**, combined with **top-5 retrieval**, ensuring:

* Enough context
* High semantic precision

---

### Embedding Generation

* Uses **Sentence-BERT (`all-MiniLM-L6-v2`)**
* Produces **384-dimensional dense vectors**
* Chosen after testing:

  * Word-level embeddings (insufficient context)
  * Character-based embeddings (weaker semantics)

---

### Vector Database (FAISS)

* FAISS `IndexFlatIP` is used
* All embeddings are **L2-normalized**
* Inner product similarity becomes **cosine similarity**
* Index is saved to disk for **one-time preprocessing**

---

### Query Processing & Retrieval

For each user question:

1. Query is embedded using the same Sentence-BERT model
2. FAISS performs cosine similarity search
3. **Top-5 most relevant chunks** are retrieved
4. Similarity scores and chunk previews are optionally printed in DEBUG mode

---

### Prompt Engineering & Generation

* Retrieved chunks are combined into a single **External Data** section
* The LLM is explicitly instructed:

  * Use only the provided context
  * Refuse if information is missing
* Temperature is set low (`0.1`) to reduce randomness

---

### Debug Mode

A `DEBUG` flag allows inspection of:

* Sample text chunks
* Embedding values
* Cosine similarity scores
* Retrieved context passed to the LLM

This improves transparency and aligns with explainable AI principles.

---

## Limitations

* Retrieval quality depends on chunking granularity
* The system is sensitive to query formulation
* Free API availability may change
* Narrative questions requiring inference may fail if not explicitly stated in text

---

## Conclusion

This project successfully demonstrates a **complete and transparent RAG system**, achieving:

* Clear separation of retrieval and generation
* Grounded, non-hallucinatory answers
* Efficient semantic search using dense embeddings
* Strong alignment with theoretical RAG concepts taught in class

The iterative design process, including failed attempts and refinements, reflects both **deep learning understanding** and **practical NLP system design**.

---

## 👤 Author

**Negar Chamanitabrez**
EEE517 – Deep Learning & NLP Project
