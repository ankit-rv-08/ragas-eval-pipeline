"""Postgres connection and schema setup for the RAGAS evaluation pipeline."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
SCHEMA = "ragas_eval"

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL not set. Copy .env.example to .env and add your database URL."
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def init_schema() -> None:
    """Create the evaluation schema and tables if they do not exist."""
    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.runs (
                    id SERIAL PRIMARY KEY,
                    config_name TEXT NOT NULL,
                    chunk_size INT NOT NULL,
                    top_k INT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
                """
            )
        )
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.results (
                    id SERIAL PRIMARY KEY,
                    run_id INT REFERENCES {SCHEMA}.runs(id) ON DELETE CASCADE,
                    question TEXT NOT NULL,
                    generated_answer TEXT NOT NULL,
                    expected_answer TEXT,
                    faithfulness FLOAT,
                    answer_relevancy FLOAT,
                    context_precision FLOAT,
                    context_recall FLOAT,
                    retrieved_sources JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
                """
            )
        )

    print(f"Schema {SCHEMA} initialized")


if __name__ == "__main__":
    init_schema()