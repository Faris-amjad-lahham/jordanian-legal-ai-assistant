import re
from typing import Dict, Any

from Legal_QA import legal_qa_chatbot
from contract_backend import ContractStore, generate_contract


GOLDEN_DATASET = [
    {"id": "Q1", "question": "مكافأة نهاية الخدمة", "expected_article": 32},
    {"id": "Q2", "question": "مكافأة نهاية الخدمه", "expected_article": 32},
    {"id": "Q3", "question": "end of service gratuity", "expected_article": 32},
    {"id": "Q4", "question": "articles 1-3", "expected_article": 1},
]


def recall_at_k(result: Dict[str, Any], expected_article: int) -> bool:
    ids = result.get("retrieved_article_ids", [])
    return expected_article in ids


def run_rag_evaluation(top_k: int = 3):
    total = len(GOLDEN_DATASET)
    hit = 0
    rows = []

    for s in GOLDEN_DATASET:
        res = legal_qa_chatbot(s["question"], top_k=top_k, return_metadata=True)
        ok = recall_at_k(res, s["expected_article"])
        hit += 1 if ok else 0
        rows.append({
            "id": s["id"],
            "question": s["question"],
            "expected": s["expected_article"],
            "retrieved": res.get("retrieved_article_ids", []),
            "pass": ok,
        })

    acc = (hit / total) * 100 if total else 0
    return {"accuracy": acc, "rows": rows}


def run_contract_evaluation():
    store = ContractStore()
    data = {
        "date": "Today",
        "contract_type": "Limited",
        "employer": "Test Co",
        "employee": "Ahmad",
        "job_title": "Engineer",
        "location": "Amman",
        "start_date": "01/01/2026",
        "end_date": "01/01/2027",
        "salary": "500",
        "hours": "8",
        "days": "5",
        "termination": "30 days notice",
    }
    text = generate_contract(store, data)

    checks = {
        "contains_employee": bool(re.search(r"\bAhmad\b", text)),
        "contains_salary": "500" in text,
        "has_disclaimer": "LEGAL DISCLAIMER:" in text,
    }

    ok = all(checks.values())
    return {"ok": ok, "checks": checks}


def print_report(rag, contract):
    print("\n================ EVALUATION REPORT ================\n")
    print(f"RAG Retrieval Recall@K Accuracy: {rag['accuracy']:.1f}%")
    for r in rag["rows"]:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"- {r['id']} {status} | expected={r['expected']} | retrieved={r['retrieved']} | q={r['question']}")

    print("\nContract Drafting Validity:", "PASS" if contract["ok"] else "FAIL")
    for k, v in contract["checks"].items():
        print(f"- {k}: {'PASS' if v else 'FAIL'}")

    print("\nDone.\n")


if __name__ == "__main__":
    rag = run_rag_evaluation(top_k=3)
    contract = run_contract_evaluation()
    print_report(rag, contract)
