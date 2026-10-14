"""Query the local vector store and print the top matching chunks.

Run with: python -m scripts.test_retrieval
"""

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUERY = "What is coronary artery disease?"


def main() -> None:
    """Run a similarity search against the local Chroma store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    results = vectorstore.similarity_search_with_score(QUERY, k=5)

    print(f"\nQuery: {QUERY}\n")
    for index, (document, score) in enumerate(results, 1):
        print(f"--- Result {index} (score: {score:.4f}) ---")
        print(
            "Source: "
            f"{document.metadata.get('source_file', '?')} "
            f"page {document.metadata.get('page', '?')}"
        )
        print(f"Chunk:  {document.page_content[:250]}...\n")


if __name__ == "__main__":
    main()