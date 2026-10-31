from app.evaluation import retrieval_metrics, token_f1


def test_retrieval_metrics_perfect():
    result = retrieval_metrics(["a", "b"], ["a", "b"])
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_retrieval_metrics_partial():
    result = retrieval_metrics(["a", "b"], ["a", "c"])
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_token_f1_identical():
    assert token_f1("bank account", "bank account") == 1.0
