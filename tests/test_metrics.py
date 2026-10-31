import pytest
from app.evaluation import evaluate_cases, retrieval_metrics, token_f1

def test_retrieval_metrics_perfect():
    assert retrieval_metrics(["a", "b"], ["a", "b"]) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

def test_retrieval_metrics_partial():
    assert retrieval_metrics(["a", "b"], ["a", "c"]) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}

@pytest.mark.parametrize(("expected", "retrieved", "p", "r", "f"), [
    ([], [], 0, 0, 0), (["a"], [], 0, 0, 0), ([], ["a"], 0, 0, 0),
    (["a", "b"], ["a"], 1, 0.5, 2 / 3),
])
def test_retrieval_metric_edge_cases(expected, retrieved, p, r, f):
    result = retrieval_metrics(expected, retrieved)
    assert result["precision"] == pytest.approx(p)
    assert result["recall"] == pytest.approx(r)
    assert result["f1"] == pytest.approx(f)

def test_token_f1_identical():
    assert token_f1("bank account", "bank account") == 1.0

def test_token_f1_no_overlap():
    assert token_f1("savings account", "credit card") == 0.0

def test_evaluate_cases_uses_fake_assistant():
    class FakeAssistant:
        def answer(self, question):
            return {"answer": "A deposit account", "retrieved_documents": [{"id": "doc_savings", "text": "A deposit account"}], "generation_mode": "test"}
    cases = [{"question": "What is a savings account?", "expected_answer": "A deposit account", "relevant_doc_ids": ["doc_savings"]}]
    result = evaluate_cases(FakeAssistant(), cases)
    assert result["cases"] == 1
    assert result["mean_retrieval_f1"] == 1.0
