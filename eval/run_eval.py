"""
Evaluation script — Measures retrieval quality of the RAG system.
Computes Hit Rate @k, MRR, and Precision @k for k ∈ {1, 3, 5}.
"""
import json
import sys
import os
import requests
from collections import defaultdict

API_URL = os.getenv("FLASK_API_URL", "http://localhost:5000")
EVAL_DATASET = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_eval_dataset():
    """Load the evaluation dataset."""
    with open(EVAL_DATASET, "r") as f:
        return json.load(f)


def query_system(question: str, top_k: int = 5):
    """Query the RAG system and return sources."""
    try:
        resp = requests.post(
            f"{API_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"  [ERROR] Query failed: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        print(f"  [ERROR] Cannot reach API: {e}")
        return None


def compute_metrics(results, k_values=[1, 3, 5]):
    """
    Compute retrieval metrics:
    - Hit Rate @k: fraction of queries where the relevant doc appears in top-k
    - MRR (Mean Reciprocal Rank): average of 1/rank of first relevant result
    - Precision @k: fraction of top-k results that are relevant
    """
    metrics = {}

    for k in k_values:
        hits = 0
        reciprocal_ranks = []
        precisions = []

        for r in results:
            expected_doc = r["expected_doc_id"]
            retrieved_docs = r.get("retrieved_doc_ids", [])[:k]

            # Hit Rate
            hit = 1 if expected_doc in retrieved_docs else 0
            hits += hit

            # Reciprocal Rank
            rr = 0.0
            for i, doc_id in enumerate(retrieved_docs):
                if doc_id == expected_doc:
                    rr = 1.0 / (i + 1)
                    break
            reciprocal_ranks.append(rr)

            # Precision @k
            relevant_count = sum(1 for d in retrieved_docs if d == expected_doc)
            precisions.append(relevant_count / k if k > 0 else 0)

        n = len(results) if results else 1
        metrics[f"hit_rate@{k}"] = round(hits / n, 4)
        metrics[f"mrr@{k}"] = round(sum(reciprocal_ranks) / n, 4)
        metrics[f"precision@{k}"] = round(sum(precisions) / n, 4)

    return metrics


def run_evaluation():
    """Run the full evaluation pipeline."""
    dataset = load_eval_dataset()
    questions = dataset["questions"]
    max_k = 5

    print(f"=== RAG Retrieval Evaluation ===")
    print(f"Total questions: {len(questions)}")
    print(f"k values: [1, 3, 5]")
    print()

    results = []
    successes = []
    failures = []

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Querying: {q['question'][:60]}...")
        response = query_system(q["question"], top_k=max_k)

        if response is None:
            print("  → Skipped (API error)")
            continue

        sources = response.get("sources", [])
        retrieved_doc_ids = [s["doc_id"] for s in sources]

        result = {
            "question_id": q["id"],
            "question": q["question"],
            "expected_doc_id": q["expected_doc_id"],
            "retrieved_doc_ids": retrieved_doc_ids,
            "answer": response.get("answer", ""),
            "num_sources": len(sources),
        }
        results.append(result)

        # Classify as success or failure
        if q["expected_doc_id"] in retrieved_doc_ids[:3]:
            successes.append(result)
            print(f"  ✅ Relevant doc found in top-3")
        else:
            failures.append(result)
            print(f"  ❌ Relevant doc NOT in top-3")

    print()

    # Compute metrics
    metrics = compute_metrics(results)

    print("=== Retrieval Metrics ===")
    for metric, value in sorted(metrics.items()):
        print(f"  {metric}: {value}")
    print()

    # Qualitative analysis
    print("=== Qualitative Analysis ===")
    print("\n--- Success Cases ---")
    for s in successes[:3]:
        print(f"  Q: {s['question']}")
        print(f"  Expected: {s['expected_doc_id']}")
        print(f"  Retrieved: {s['retrieved_doc_ids'][:3]}")
        print(f"  Answer preview: {s['answer'][:150]}...")
        print()

    print("--- Failure Cases ---")
    for f in failures[:2]:
        print(f"  Q: {f['question']}")
        print(f"  Expected: {f['expected_doc_id']}")
        print(f"  Retrieved: {f['retrieved_doc_ids'][:3]}")
        print(f"  Explanation: The retrieval system failed to rank the relevant document")
        print(f"    in the top results. This may be due to vocabulary mismatch between")
        print(f"    the query and the stored chunks, or insufficient semantic overlap.")
        print()

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "metrics": metrics,
        "total_questions": len(questions),
        "questions_evaluated": len(results),
        "success_count": len(successes),
        "failure_count": len(failures),
        "detailed_results": results,
        "qualitative_analysis": {
            "successes": successes[:3],
            "failures": failures[:2],
        },
    }

    output_path = os.path.join(RESULTS_DIR, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    run_evaluation()
