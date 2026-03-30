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


def build_doc_id_map(dataset):
    """
    Query the API for uploaded documents and map each eval filename
    to its real UUID doc_id assigned by the system.
    Returns a dict: expected_doc_id (e.g. 'eval_doc_1') -> real UUID.
    """
    filename_to_eval_id = {
        d["filename"]: d["doc_id"] for d in dataset.get("documents", [])
    }
    try:
        resp = requests.get(f"{API_URL}/documents", timeout=10)
        if resp.status_code != 200:
            print("[WARN] Could not fetch document list — doc_id matching may be incorrect")
            return {}
        uploaded = resp.json().get("documents", [])
    except Exception as e:
        print(f"[WARN] Could not reach API for doc mapping: {e}")
        return {}

    eval_id_to_uuid = {}
    for doc in uploaded:
        eval_id = filename_to_eval_id.get(doc["filename"])
        if eval_id:
            eval_id_to_uuid[eval_id] = doc["id"]

    print("Document ID mapping (eval_id -> system UUID):")
    for eid, uid in eval_id_to_uuid.items():
        print(f"  {eid} -> {uid}")
    if not eval_id_to_uuid:
        print("  [WARN] No eval documents found. Upload them first.")
    print()
    return eval_id_to_uuid


def run_evaluation():
    """Run the full evaluation pipeline."""
    dataset = load_eval_dataset()
    questions = dataset["questions"]
    k_values = [1, 3, 5]

    print("=== RAG Retrieval Evaluation ===")
    print(f"Total questions: {len(questions)}")
    print(f"k values: {k_values}")
    print()

    # Resolve symbolic eval doc_ids to real UUIDs assigned by the API
    doc_id_map = build_doc_id_map(dataset)

    # ── Step 1: Query each question once with top_k=5 ────────────────────────
    print("── Querying system (top_k=5) ──")
    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['question'][:65]}...")
        response = query_system(q["question"], top_k=5)

        if response is None:
            print("  → Skipped (API error)")
            continue

        sources = response.get("sources", [])
        retrieved_doc_ids = [s["doc_id"] for s in sources]

        expected_eval_id = q["expected_doc_id"]
        expected_real_id = doc_id_map.get(expected_eval_id, expected_eval_id)

        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "expected_doc_id": expected_real_id,
            "retrieved_doc_ids": retrieved_doc_ids,
            "answer": response.get("answer", ""),
            "num_sources": len(sources),
        })
    print()

    # ── Step 2: Show pass/fail for each k ────────────────────────────────────
    for k in k_values:
        print(f"── Results at k={k} ──")
        successes_k = []
        failures_k = []
        for r in results:
            hit = r["expected_doc_id"] in r["retrieved_doc_ids"][:k]
            icon = "✅" if hit else "❌"
            print(f"  {icon} [{r['question_id']}] {r['question'][:60]}")
            if hit:
                successes_k.append(r)
            else:
                failures_k.append(r)
        print(f"  → Hits: {len(successes_k)}/{len(results)}")
        print()

    # ── Step 3: Compute and print all metrics ─────────────────────────────────
    metrics = compute_metrics(results, k_values)

    print("=== Retrieval Metrics ===")
    for k in k_values:
        print(f"  k={k}:  Hit Rate={metrics[f'hit_rate@{k}']}  "
              f"MRR={metrics[f'mrr@{k}']}  "
              f"Precision={metrics[f'precision@{k}']}")
    print()

    # ── Step 4: Qualitative analysis (based on k=3) ───────────────────────────
    successes = [r for r in results if r["expected_doc_id"] in r["retrieved_doc_ids"][:3]]
    failures  = [r for r in results if r["expected_doc_id"] not in r["retrieved_doc_ids"][:3]]

    print("=== Qualitative Analysis (k=3) ===")
    print("\n--- Success Cases ---")
    for s in successes[:3]:
        print(f"  Q: {s['question']}")
        print(f"  Retrieved top-3: {s['retrieved_doc_ids'][:3]}")
        print(f"  Answer preview: {s['answer'][:150]}...")
        print()

    print("--- Failure Cases ---")
    for f in failures[:2]:
        print(f"  Q: {f['question']}")
        print(f"  Expected: {f['expected_doc_id']}")
        print(f"  Retrieved top-3: {f['retrieved_doc_ids'][:3]}")
        print("  Explanation: The retrieval system failed to rank the relevant document")
        print("    in the top results. This may be due to vocabulary mismatch between")
        print("    the query and the stored chunks, or insufficient semantic overlap.")
        print()

    # ── Step 5: Save results ──────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "metrics": metrics,
        "total_questions": len(questions),
        "questions_evaluated": len(results),
        "success_count_k3": len(successes),
        "failure_count_k3": len(failures),
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
