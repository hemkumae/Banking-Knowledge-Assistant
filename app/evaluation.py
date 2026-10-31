import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from app.assistant import BankingAssistant
from app.config import OPENAI_API_KEY, OPENAI_MODEL

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data" / "golden_dataset.json"
RESULTS_PATH = ROOT / "evaluation_results.json"


def retrieval_metrics(expected_ids, retrieved_ids):
    """Calculate set-based Precision, Recall and F1 for one query."""
    expected, retrieved = set(expected_ids), set(retrieved_ids)
    true_positives = len(expected & retrieved)
    precision = true_positives / len(retrieved) if retrieved else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def token_f1(prediction, reference):
    """Lexical token-overlap baseline; this does not measure factuality."""
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


def judge_answer(question: str, answer: str, context: str, expected_answer: str) -> dict[str, Any]:
    """Optional model-based review. Scores are model judgments, not ground truth."""
    if not OPENAI_API_KEY:
        return {"enabled": False, "reason": "OPENAI_API_KEY is not configured"}

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer_evaluation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "grounded": {"type": "boolean"},
                        "relevance_score": {"type": "integer"},
                        "correctness_score": {"type": "integer"},
                        "rationale": {"type": "string"}
                    },
                    "required": ["grounded", "relevance_score", "correctness_score", "rationale"],
                    "additionalProperties": False
                }
            }
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a cautious evaluator of a banking-domain RAG answer. "
                    "Use only the supplied retrieved context and expected answer as references. "
                    "Do not treat the expected answer as proof if the context contradicts it. "
                    "Score relevance and correctness from 1 (poor) to 5 (excellent). "
                    "Grounded is true only when the answer's factual claims are supported by context."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nExpected answer: {expected_answer}"
                    f"\nRetrieved context: {context}\nCandidate answer: {answer}"
                )
            }
        ]
    )
    message = response.choices[0].message
    if not message.content:
        raise ValueError("The evaluator returned no structured content")
    result = json.loads(message.content)
    if not 1 <= result["relevance_score"] <= 5 or not 1 <= result["correctness_score"] <= 5:
        raise ValueError("Judge scores must be between 1 and 5")
    return {"enabled": True, **result}


def evaluate_cases(assistant, cases, use_llm_judge=False):
    rows = []
    for case in cases:
        result = assistant.answer(case["question"])
        documents = result["retrieved_documents"]
        retrieved_ids = [doc["id"] for doc in documents]
        context = "\n".join(doc["text"] for doc in documents)
        row = {
            "question": case["question"],
            **retrieval_metrics(case["relevant_doc_ids"], retrieved_ids),
            "answer_token_f1": token_f1(result["answer"], case["expected_answer"]),
            "retrieved_ids": retrieved_ids,
            "generation_mode": result.get("generation_mode", "unknown"),
        }
        if use_llm_judge:
            row["llm_judge"] = judge_answer(
                case["question"], result["answer"], context, case["expected_answer"]
            )
        rows.append(row)

    count = len(rows)
    return {
        "cases": count,
        "mean_precision": sum(row["precision"] for row in rows) / count if count else 0.0,
        "mean_recall": sum(row["recall"] for row in rows) / count if count else 0.0,
        "mean_retrieval_f1": sum(row["f1"] for row in rows) / count if count else 0.0,
        "mean_answer_token_f1": sum(row["answer_token_f1"] for row in rows) / count if count else 0.0,
        "llm_judge_requested": use_llm_judge,
        "details": rows,
    }


def run_evaluation(use_llm_judge=False, output_path=RESULTS_PATH):
    cases = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    summary = evaluate_cases(BankingAssistant(), cases, use_llm_judge=use_llm_judge)
    Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the banking RAG assistant.")
    parser.add_argument("--llm-judge", action="store_true", help="Enable optional OpenAI model judging.")
    args = parser.parse_args()
    run_evaluation(use_llm_judge=args.llm_judge)
