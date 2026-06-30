from util import PDFReader, Chunker, TextEmbedder, VectorDB

# ---------------------------
# CONFIG
# ---------------------------
PDF_PATH = "./example.pdf"
COLLECTION_NAME = "pdf_rag"


def build_index():
    print("\n[INDEXING] Starting document ingestion...")

    # load components
    reader = PDFReader(PDF_PATH)
    chunker = Chunker()
    embedder = TextEmbedder()
    db = VectorDB()

    # step 1: extract text
    text = reader.get_text()

    # step 2: chunk text
    chunks = chunker.chunk_text(text)
    print(f"[INFO] Created {len(chunks)} chunks")

    # step 3: create embeddings
    embeddings = [embedder.embed(c) for c in chunks]

    # step 4: store in vector DB
    collection = db.get_collection(COLLECTION_NAME)

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

    print("[DONE] Vector DB built successfully.")


if __name__ == "__main__":
    build_index()