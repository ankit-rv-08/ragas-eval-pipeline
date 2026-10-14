"""Ingest PDFs into a Chroma vector store.

Usage: python -m src.ingest
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

CORPUS_DIR = Path("data/corpus")
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdfs(corpus_dir: Path):
    """Load all PDFs from a directory and tag each page with its source file."""
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {corpus_dir}. Add at least one .pdf file."
        )

    documents = []
    for pdf in tqdm(pdfs, desc="Loading PDFs"):
        pages = PyPDFLoader(str(pdf)).load()
        for page in pages:
            page.metadata["source_file"] = pdf.name
        documents.extend(pages)
    return documents


def chunk_documents(documents, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks, persist_dir=CHROMA_DIR):
    """Embed chunks and store them in Chroma."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )


def main() -> None:
    """Run the PDF ingestion pipeline."""
    print(f"Loading PDFs from {CORPUS_DIR}...")
    documents = load_pdfs(CORPUS_DIR)
    print(f"Loaded {len(documents)} pages")

    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Embedding and storing in Chroma...")
    build_vectorstore(chunks)
    print(f"Vector store saved to {CHROMA_DIR}")


if __name__ == "__main__":
    main()