# Banking Knowledge Assistant

A portfolio project demonstrating a small Retrieval-Augmented Generation (RAG) system, a banking knowledge graph, an evaluation harness, and REST APIs. All banking data is illustrative.

## Features
- Sentence Transformers embeddings and FAISS similarity retrieval.
- Optional OpenAI-generated answers; without an API key, the assistant returns retrieved context.
- NetworkX directed graph with typed entities and labelled relationships.
- Golden-dataset retrieval evaluation using Precision, Recall and F1.
- Optional schema-constrained LLM-as-a-judge for groundedness, relevance and correctness.
- pytest coverage for metrics, graph operations and API endpoints.
- GitHub Actions workflow to run tests on pushes and pull requests.

## Setup
Use Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Add OPENAI_API_KEY to .env only if you want generated answers and model judging. Never commit .env or API keys.

## Run the API
```bash
uvicorn app.api:app --reload
```
Open http://127.0.0.1:8000/docs.

Endpoints:
- GET /health
- POST /ask
- GET /graph/entities
- GET /graph/relationships?relation=OPENS
- GET /graph/entity/Customer?direction=outgoing
- GET /graph/path?source=Customer&target=Interest&directed=true

## Tests
```bash
pytest -q
```

## Evaluation
```bash
python -m app.evaluation
```
This writes evaluation_results.json locally. To enable the optional LLM judge:
```bash
python -m app.evaluation --llm-judge
```
The judge returns grounded (boolean), relevance_score (1-5), correctness_score (1-5), and rationale. These are model judgments, not ground truth, and require human review. The call may incur API costs.

Precision = relevant retrieved documents / all retrieved documents.
Recall = relevant retrieved documents / all expected relevant documents.
F1 = harmonic mean of Precision and Recall.

The answer token F1 is lexical overlap only; it does not prove factual correctness or faithfulness. The LLM judge is fallible and should be checked against human-labelled examples.

## Limitations
Educational portfolio software, not a production banking system. No real customer data, banking integration, regulatory controls, or financial advice. Calibration curves, automated entity extraction, production vector database service, cloud deployment, and a full agent workflow are not implemented yet.
