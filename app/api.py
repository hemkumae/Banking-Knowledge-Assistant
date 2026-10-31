from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.assistant import BankingAssistant
from app.knowledge_graph import BankingKnowledgeGraph

app = FastAPI(title="Banking Knowledge Assistant", version="1.1.0")
assistant = None
knowledge_graph = BankingKnowledgeGraph()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


@app.on_event("startup")
def startup():
    global assistant
    assistant = BankingAssistant()


@app.get("/health")
def health():
    return {"status": "ok", "assistant_initialized": assistant is not None}


@app.post("/ask")
def ask(request: AskRequest):
    if assistant is None:
        raise HTTPException(status_code=503, detail="Assistant is still initializing")
    return assistant.answer(request.question, k=request.top_k)


@app.get("/graph/entities")
def graph_entities(entity_type: str | None = None):
    return {"entities": knowledge_graph.entities(entity_type)}


@app.get("/graph/relationships")
def graph_relationships(relation: str | None = None):
    return {"relationships": knowledge_graph.relationships(relation)}


@app.get("/graph/entity/{entity_name}")
def graph_entity(entity_name: str, direction: str = Query(default="both")):
    try:
        return knowledge_graph.neighbors(entity_name, direction=direction)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/graph/path")
def graph_path(source: str, target: str, directed: bool = True):
    return knowledge_graph.shortest_path(source, target, directed=directed)
