# RAGAS Eval Pipeline

RAG evaluation pipeline using RAGAS to compare retrieval configurations on a domain corpus, with a Streamlit dashboard.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your `GROQ_API_KEY` and Postgres `DATABASE_URL` to `.env`.

## Add a corpus

Place one or more PDFs in `data/corpus/`, then build the local Chroma vector store:

```bash
python -m src.ingest
```

Initialize the Postgres schema with:

```bash
python -m src.db
```

## Dashboard

Start the dashboard shell with:

```bash
streamlit run src/dashboard.py
```

Evaluation execution will be added in `src/evaluate.py`; the dashboard currently shows the metric layout and expected command.
