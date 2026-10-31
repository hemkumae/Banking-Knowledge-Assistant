import json
import re
from collections import Counter
from pathlib import Path

from app.assistant import BankingAssistant

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data" / "golden_dataset.json"


def retrieval_metrics(expected_ids, retrieved_ids):
    expected, retrieved = set(expected_ids), set(retrieved_ids)
    true_positives = len(expected & retrieved)
    precision = true_positives / len(retrieved) if retrieved else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def token_f1(prediction, reference):
    predicted_tokens = re.findall(r"\w+", prediction.lower())
    reference_tokens = re.findall(r"\w+", reference.lower())
    if not predicted_tokens or not reference_tokens:
        return 0.0
    common = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(predicted_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def run_evaluation():
    assistant = BankingAssistant()
    cases = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        result = assistant.answer(case["question"])
        retrieved_ids = [doc["id"] for doc in result["retrieved_documents"]]
        metrics = retrieval_metrics(case["relevant_doc_ids"], retrieved_ids)
        rows.append({
            "question": case["question"],
            **metrics,
            "answer_token_f1": token_f1(
                result["answer"], case["expected_answer"]
            ),
            "retrieved_ids": retrieved_ids,
        })

    count = len(rows) or 1
    summary = {
        "cases": len(rows),
        "mean_precision": sum(row["precision"] for row in rows) / count,
        "mean_recall": sum(row["recall"] for row in rows) / count,
        "mean_retrieval_f1": sum(row["f1"] for row in rows) / count,
        "mean_answer_token_f1": sum(
            row["answer_token_f1"] for row in rows
        ) / count,
        "details": rows,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_evaluation()
