# Chat with Your Own Text File (RAG with LangChain + Ollama)

A simple Retrieval-Augmented Generation (RAG) system that allows you to chat with your own documents using a local LLM.

---

## 🚀 Features
- Chat with custom text files
- Local LLM (Ollama - no API cost)
- FAISS vector search
- Semantic retrieval using embeddings
- Fully offline AI pipeline

---

## 🧠 Tech Stack
- Python
- LangChain
- Ollama (llama3)
- FAISS

---

## ⚙️ How it works
1. Load text file
2. Split into chunks
3. Convert to embeddings
4. Store in vector DB
5. Retrieve relevant chunks
6. Send to LLM
7. Generate answer

---

## ▶️ Run

```bash
pip install -r requirements.txt
python3 main.py