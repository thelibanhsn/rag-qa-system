import os
from dotenv import load_dotenv
from rag_engine import RAGEngine

load_dotenv()


def main():
    print("\nRAG System Started (type 'exit' to quit)\n")

    engine = RAGEngine(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=os.getenv("API_URL"),
        model_name=os.getenv("GROQ_MODEL"),
        collection_name="pdf_rag"
    )

    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            break

        answer = engine.answer(query)

        print("\nAgent:", answer, "\n")


if __name__ == "__main__":
    main()