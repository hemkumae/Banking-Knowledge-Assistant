from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.assistant import BankingAssistant
from app.knowledge_graph import BankingKnowledgeGraph

app = FastAPI(title="Banking Knowledge Assistant", version="1.0.0")
assistant = None
knowledge_graph = BankingKnowledgeGraph()


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=3, ge=1, le=10)


@app.on_event("startup")
def startup():
    global assistant
    assistant = BankingAssistant()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    return assistant.answer(request.question, k=request.top_k)


@app.get("/graph/entity/{entity_name}")
def graph_entity(entity_name: str):
    return knowledge_graph.neighbors(entity_name)


@app.get("/graph/path")
def graph_path(source: str, target: str):
    return knowledge_graph.shortest_path(source, target)
