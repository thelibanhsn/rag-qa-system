from util import TextEmbedder, VectorDB, LLMClient


class RAGEngine:
    def __init__(self, api_key, base_url, model_name, collection_name):
        self.embedder = TextEmbedder()
        self.db = VectorDB()
        self.collection = self.db.get_collection(collection_name)

        self.llm = LLMClient(api_key, base_url)
        self.model = model_name

        # retrieval tuning
        self.top_k = 5
        self.max_distance = 0.8

    # ---------------------------
    # RETRIEVE RELEVANT CHUNKS
    # ---------------------------
    def retrieve(self, query):
        query_embedding = self.embedder.embed(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )

        docs = []
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            if dist <= self.max_distance:
                docs.append(doc)

        return docs

    # ---------------------------
    # BUILD PROMPT
    # ---------------------------
    def build_prompt(self, query, docs):
        context = "\n\n".join(docs)

        return f"""
            You are a strict RAG assistant.

            Answer ONLY using the context below.
            If the answer is not in the context, say "I don't know".

            Context:
            {context}

            Question:
            {query}
        """

    # ---------------------------
    # GENERATE ANSWER
    # ---------------------------
    def answer(self, query):
        docs = self.retrieve(query)

        if not docs:
            return "I don't know based on the provided documents."

        prompt = self.build_prompt(query, docs)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        return self.llm.generate(self.model, messages)