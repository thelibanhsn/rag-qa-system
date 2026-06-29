from util import *
import chromadb

# create a ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

collection_name = "pdf_embeddings"

# create or get the collection
collection = client.get_or_create_collection(name=collection_name)



# read the file and extract text
reader = PDFreader("./example.pdf")
text = reader.get_text()

# chunk the text into smaller pieces
chunker = Chunker(text)
chunks = chunker.chunk_text()

# embed the chunks of text
embedder = TextEmbedder()
embeddings = [embedder.embed_text(chunk) for chunk in chunks]

print(f" Text extracted from PDF: {text[:100]}...")  # Print first 100 characters of the text
print(f" Number of chunks created: {len(chunks)}")
print(f" Embeddings shape: {len(embeddings)} chunks with embedding size {len(embeddings[0]) if embeddings else 0}")

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)


data = collection.get()
print(f"Retrieved {len(data['ids'])} items from the collection.")
print(f"First item ID: {data['ids'][0]}")
print(f"First item document: {data['documents'][0][:100]}...")  # Print first 100 characters of the first document

