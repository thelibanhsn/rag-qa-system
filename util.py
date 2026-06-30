from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import os


# ---------------------------
# PDF READER
# ---------------------------
class PDFReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.reader = PdfReader(file_path)

    def get_text(self):
        """Extract text safely from all PDF pages"""
        text = ""
        for page in self.reader.pages:
            text += (page.extract_text() or "")
        return text


# ---------------------------
# TEXT CHUNKING
# ---------------------------
class Chunker:
    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        """Simple sliding window chunking"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap

        return chunks


# ---------------------------
# EMBEDDINGS MODEL (loaded once)
# ---------------------------
class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text):
        return self.model.encode(text)


# ---------------------------
# CHROMA DB CLIENT WRAPPER
# ---------------------------
class VectorDB:
    def __init__(self, path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)

    def get_collection(self, name):
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}  
        )

    def delete_collection(self, name):
        try:
            self.client.delete_collection(name)
        except Exception:
            pass  # ignore if not exists


# ---------------------------
# LLM CLIENT (Groq / OpenAI compatible)
# ---------------------------
class LLMClient:
    def __init__(self, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate(self, model, messages):
        """Call LLM safely and return text only"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2
        )

        return response.choices[0].message.content