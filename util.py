from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

class PDFreader:
    def __init__(self, file_path):
        print(f"Reading PDF file: {file_path}")
        self.file_path = file_path
        self.reader = PdfReader(file_path)

    def get_text(self):
        text = ""
        for page in self.reader.pages:
            text += page.extract_text()
        return text

class Chunker:
    def __init__(self, text, chunk_size=500, overlap=100):
        self.text = text
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self):
        chunks = []
        for i in range(0, len(self.text), self.chunk_size - self.overlap):
            chunks.append(self.text[i:i + self.chunk_size])
        return chunks

class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text):
        return self.model.encode(text)