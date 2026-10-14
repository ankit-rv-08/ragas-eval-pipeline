"""RAG pipeline: retrieve Chroma chunks and generate cited answers with Groq."""

import re
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from groq import BadRequestError
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are a clinical information assistant. Answer the user's question using ONLY the provided context.

Rules:
- If the context does not contain the answer, say "I don't have enough information in the provided documents to answer that."
- Cite sources inline using [Source N] notation where N is the source number.
- Be concise and accurate. Do not invent facts.
- Do not use outside knowledge.

Context:
{context}
"""

USER_PROMPT = "{question}"


@dataclass
class RetrievedChunk:
    content: str
    source_file: str
    page: int
    score: float
    rank: int


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class RAGPipeline:
    def __init__(
        self,
        top_k: int = 5,
        persist_dir: str = CHROMA_DIR,
        embed_model: str = EMBED_MODEL,
        llm_model: str = LLM_MODEL,
    ):
        self.top_k = top_k
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )
        self.llm = ChatGroq(
            model=llm_model,
            temperature=0.1,
            max_tokens=1024,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
        )
        self.chain = self.prompt | self.llm

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        results = self.vectorstore.similarity_search_with_score(
            question, k=self.top_k
        )
        return [
            RetrievedChunk(
                content=document.page_content,
                source_file=document.metadata.get("source_file", "unknown"),
                page=document.metadata.get("page", -1),
                score=float(score),
                rank=rank,
            )
            for rank, (document, score) in enumerate(results, 1)
        ]

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        parts = []
        for chunk in chunks:
            parts.append(
                f"[Source {chunk.rank}] ({chunk.source_file}, page {chunk.page + 1}):\n"
                f"{chunk.content}"
            )
        return "\n\n".join(parts)

    def _invoke_with_retry(
        self, inputs: dict[str, str], max_retries: int = 5, base_delay: int = 20
    ):
        for attempt in range(max_retries):
            try:
                return self.chain.invoke(inputs)
            except BadRequestError as error:
                is_rate_limited = getattr(error, "status_code", None) == 429 or (
                    "429" in str(error) or "rate" in str(error).lower()
                )
                if not is_rate_limited or attempt == max_retries - 1:
                    raise
                delay = base_delay * (2**attempt)
                print(
                    f"Rate limited, retrying in {delay}s "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(delay)

    def answer(self, question: str) -> RAGResult:
        start = time.perf_counter()
        chunks = self.retrieve(question)
        response = self._invoke_with_retry(
            {
                "context": self.format_context(chunks),
                "question": question,
            }
        )
        latency_ms = (time.perf_counter() - start) * 1000
        answer = re.sub(r"【Source\s+(\d+)】", r"[Source \1]", response.content)
        usage = getattr(response, "usage_metadata", {}) or {}
        if not usage:
            usage = getattr(response, "response_metadata", {}).get("token_usage", {})

        return RAGResult(
            question=question,
            answer=answer,
            sources=chunks,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("input_tokens", usage.get("prompt_tokens")),
            completion_tokens=usage.get(
                "output_tokens", usage.get("completion_tokens")
            ),
        )


if __name__ == "__main__":
    rag = RAGPipeline()
    questions = [
        "What is coronary artery disease?",
        "What ECG findings are associated with CAD?",
        "What are the major risk factors for coronary artery disease?",
    ]
    for question in questions:
        print("=" * 70)
        print(f"Q: {question}")
        print("=" * 70)
        result = rag.answer(question)
        print(f"\nA: {result.answer}\n")
        print(f"Latency: {result.latency_ms:.0f} ms")
        print(
            "Tokens: "
            f"prompt={result.prompt_tokens}, "
            f"completion={result.completion_tokens}"
        )
        print(f"Sources ({len(result.sources)}):")
        for source in result.sources:
            print(
                f"  [{source.rank}] {source.source_file} "
                f"p.{source.page + 1} (score: {source.score:.4f})"
            )
        print()