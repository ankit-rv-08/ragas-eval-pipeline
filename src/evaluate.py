"""Evaluate the RAG pipeline against the checked-in test set with RAGAS.

Usage: python -m src.evaluate
"""

import json
import os
import sys
import time
import types
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from groq import Groq
from sqlalchemy import text

from src.rag import RAGPipeline

load_dotenv()

TEST_SET_PATH = Path("tests/test_set.json")
SCHEMA = "ragas_eval"
CONFIG_NAME = "chunk_800_topk_5"
CHUNK_SIZE = 800
TOP_K = 5
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
JUDGE_MODEL = "openai/gpt-oss-120b"


def _install_ragas_compatibility_shim() -> None:
    """Restore the import path removed from recent langchain-community releases."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[module_name] = module


def load_test_set() -> list[dict]:
    """Load and return the evaluation questions."""
    with TEST_SET_PATH.open() as file:
        return json.load(file)


def create_run(engine, config_name: str, chunk_size: int, top_k: int) -> int:
    """Insert a run row and return its database id."""
    with engine.begin() as connection:
        result = connection.execute(
            text(
                f"""
                INSERT INTO {SCHEMA}.runs (config_name, chunk_size, top_k)
                VALUES (:config_name, :chunk_size, :top_k)
                RETURNING id
                """
            ),
            {
                "config_name": config_name,
                "chunk_size": chunk_size,
                "top_k": top_k,
            },
        )
        return result.scalar_one()


def save_result(engine, run_id: int, row: dict, metrics: dict) -> None:
    """Insert one evaluated question and its retrieved sources."""
    sources_json = json.dumps(
        [
            {
                "content": source.content[:500],
                "source_file": source.source_file,
                "page": source.page,
                "score": source.score,
                "rank": source.rank,
            }
            for source in row["sources"]
        ]
    )

    with engine.begin() as connection:
        connection.execute(
            text(
                f"""
                INSERT INTO {SCHEMA}.results
                    (run_id, question, generated_answer, expected_answer,
                     faithfulness, answer_relevancy, context_precision,
                     context_recall, retrieved_sources)
                VALUES
                    (:run_id, :question, :answer, :expected,
                     :faithfulness, :answer_relevancy, :context_precision,
                     :context_recall, CAST(:sources AS JSONB))
                """
            ),
            {
                "run_id": run_id,
                "question": row["question"],
                "answer": row["answer"],
                "expected": row["expected_answer"],
                "faithfulness": metrics.get("faithfulness"),
                "answer_relevancy": metrics.get("answer_relevancy"),
                "context_precision": metrics.get("context_precision"),
                "context_recall": metrics.get("context_recall"),
                "sources": sources_json,
            },
        )


def main() -> None:
    """Run RAGAS evaluation and save per-question results to Postgres."""
    _install_ragas_compatibility_shim()
    from ragas import evaluate
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
    from ragas.metrics.collections.context_precision import ContextPrecision
    from ragas.metrics.collections.context_recall import ContextRecall
    from ragas.metrics.collections.faithfulness import Faithfulness

    from src.db import engine

    started = time.perf_counter()
    print("Loading test set...")
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} questions")

    print("Initializing RAG pipeline...")
    rag = RAGPipeline(top_k=TOP_K)

    print("Running RAG on all questions...")
    rows = []
    for index, item in enumerate(test_set, 1):
        print(f"  [{index}/{len(test_set)}] {item['question'][:60]}...")
        result = rag.answer(item["question"])
        rows.append(
            {
                "question": item["question"],
                "answer": result.answer,
                "expected_answer": item["ground_truth"],
                "contexts": [chunk.content for chunk in result.sources],
                "sources": result.sources,
            }
        )

    print("\nRunning RAGAS evaluation...")
    judge = llm_factory(
        JUDGE_MODEL,
        provider="openai",
        client=Groq(api_key=os.environ["GROQ_API_KEY"]),
        temperature=0.0,
    )
    embeddings = RagasHuggingFaceEmbeddings(model=EMBED_MODEL)
    faithfulness_metric = Faithfulness(llm=judge)
    answer_relevancy_metric = AnswerRelevancy(llm=judge, embeddings=embeddings)
    context_precision_metric = ContextPrecision(llm=judge)
    context_recall_metric = ContextRecall(llm=judge)
    dataset = Dataset.from_list(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["expected_answer"],
            }
            for row in rows
        ]
    )
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness_metric,
            answer_relevancy_metric,
            context_precision_metric,
            context_recall_metric,
        ],
        llm=judge,
        embeddings=embeddings,
        raise_exceptions=True,
    )

    print("\nOverall metrics:")
    print(scores)

    print("\nSaving to Postgres...")
    run_id = create_run(engine, CONFIG_NAME, CHUNK_SIZE, TOP_K)
    print(f"Created run_id={run_id}")

    per_question = scores.to_pandas().to_dict("records")
    for row, score_row in zip(rows, per_question):
        save_result(engine, run_id, row, score_row)

    elapsed = time.perf_counter() - started
    print(f"\nSaved {len(rows)} results to {SCHEMA}.results")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
