# Banking Knowledge Assistant: RAG Evaluation + Knowledge Graph

A Python portfolio starter demonstrating embedding-based retrieval, a small banking knowledge graph, retrieval evaluation metrics, and REST APIs.

## Architecture
- `data/banking_docs.json` contains illustrative banking documents.
- Sentence Transformers creates document/query embeddings.
- FAISS retrieves the most similar documents.
- The assistant uses retrieved context; an OpenAI API key enables generated answers, otherwise it returns retrieved context.
- `app/evaluation.py` measures retrieval Precision, Recall, F1, and a simple answer token-overlap F1 baseline.
- NetworkX stores a small manually seeded graph of banking entities and relationships.
- FastAPI exposes question-answering and graph query endpoints.

## Setup
Use Python 3.10 or 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Add an optional `OPENAI_API_KEY` to `.env` to enable generated answers. Never commit `.env` or API keys.

## Run

```bash
uvicorn app.api:app --reload
```

Open `http://127.0.0.1:8000/docs`.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/ask \\
  -H 'Content-Type: application/json' \\
  -d '{"question":"What is KYC used for?","top_k":3}'
```

Run the evaluation harness with:

```bash
python -m app.evaluation
```

## Evaluation notes
The golden dataset stores questions, expected answers, and relevant document IDs. Retrieval Precision/Recall/F1 compare retrieved IDs with expected IDs. `answer_token_f1` is only a lexical-overlap baseline, not a measure of factual correctness or faithfulness.

## Limitations
This is a learning/portfolio baseline, not a production banking system. Documents and graph data are illustrative. It does not yet implement an LLM-as-judge, calibration analysis, automated entity extraction, Ragas/DeepEval, production security, or compliance controls. Do not use real customer data or claim production accuracy.