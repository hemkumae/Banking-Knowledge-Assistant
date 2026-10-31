from fastapi.testclient import TestClient
from app import api

class FakeAssistant:
    def answer(self, question, k=3):
        return {"question": question, "answer": "Test answer", "retrieved_documents": []}

def test_health_endpoint():
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ask_endpoint(monkeypatch):
    monkeypatch.setattr(api, "assistant", FakeAssistant())
    response = TestClient(api.app).post("/ask", json={"question": "What is KYC?", "top_k": 2})
    assert response.status_code == 200
    assert response.json()["answer"] == "Test answer"

def test_ask_validates_question():
    response = TestClient(api.app).post("/ask", json={"question": "?", "top_k": 2})
    assert response.status_code == 422

def test_graph_entities_endpoint():
    response = TestClient(api.app).get("/graph/entities?entity_type=Product")
    assert response.status_code == 200
    assert response.json()["entities"]
