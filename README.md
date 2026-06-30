# 📚 RAG Document Q&A System

A simple Retrieval-Augmented Generation (RAG) system that allows you to upload a PDF document and ask questions based only on its content.

The system chunks the document, creates embeddings, stores them in a vector database, retrieves relevant context, and uses an LLM to generate accurate answers.

---

## 🚀 Features

- Load and read PDF documents
- Split text into overlapping chunks
- Generate embeddings using SentenceTransformer
- Store embeddings in ChromaDB (persistent vector database)
- Perform semantic search using cosine similarity
- Retrieve top-k relevant chunks
- Generate answers using Groq/OpenAI-compatible LLM
- Prevent hallucination by restricting answers to document context

---

## 🏗️ Architecture

PDF → Chunking → Embeddings → ChromaDB
User Query → Embedding → Retrieval → LLM → Answer

---

## 📦 Tech Stack

- Python
- SentenceTransformers
- ChromaDB
- PyPDF
- Groq API (OpenAI-compatible)
- Requests / OpenAI SDK

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
2. Add environment variables

Create a .env file:

GROQ_API_KEY=your_api_key
API_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-120b

3. Build vector database
python index.py

4. Run the agent
python agent.py

💬 Example Usage
You: what is machine learning?
Agent: Machine Learning is a subset of AI that allows systems to learn from data...

You: who is messi?
Agent: I don't know based on the provided documents.

🧠 Key Concepts Learned
Text chunking strategies
Embedding generation
Vector similarity search
RAG (Retrieval Augmented Generation)
Prompt engineering with context
LLM integration

📌 Project Goal

To build a simple but production-style RAG system that answers questions strictly from a provided document while avoiding hallucinations.
```
