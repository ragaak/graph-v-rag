"""Eval command: python -m graph_vlm_rag eval"""

import json
from pathlib import Path


def run_evaluation(questions_file: str = "data/eval_questions.json") -> str:
    """
    Run evaluation against stored questions.

    For each question, calls answer_query() and checks if any
    expected keyword appears in the answer (case-insensitive).

    Returns:
        Summary report string
    """
    from .query import answer_query

    path = Path(questions_file)

    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    with open(path) as f:
        questions = json.load(f)

    if not questions:
        return " eval: 0 questions (empty file)"

    results = []
    passed = 0

    for i, q in enumerate(questions, 1):
        question_text = q.get("question", "")
        expected_keywords = q.get("expected_keywords", [])

        if not question_text:
            continue

        print(f"📝 [{i}/{len(questions)}] {question_text[:60]}...")

        try:
            # Get answer from query pipeline
            answer_full = answer_query(question_text)
            # Extract just the answer portion (after "A: ")
            answer_text = answer_full
            if "A: " in answer_full:
                answer_text = answer_full.split("A: ", 1)[1]
                if "[Sources:" in answer_text:
                    answer_text = answer_text.split("[Sources:")[0]

            answer_lower = answer_text.lower()

            # Check for keyword matches
            matched_keywords = [k for k in expected_keywords if k.lower() in answer_lower]

            if expected_keywords and not matched_keywords:
                status = "❌ FAIL"
            elif not expected_keywords:
                # No keywords to check - pass if answer is non-empty
                status = "✅ PASS" if answer_text.strip() else "⚠️ EMPTY"
            else:
                status = "✅ PASS"
                passed += 1

            results.append({
                "question": question_text,
                "status": status,
                "expected_keywords": expected_keywords,
                "matched": matched_keywords,
                "answer_preview": answer_text[:100].strip(),
            })

        except Exception as e:
            results.append({
                "question": question_text,
                "status": f"❌ ERROR",
                "error": str(e),
            })

    # Format report
    report_lines = []
    report_lines.append("\n" + "=" * 70)
    report_lines.append(" EVALUATION REPORT")
    report_lines.append("=" * 70)

    for i, r in enumerate(results, 1):
        report_lines.append(f"\n[{i}] {r['status']} {r['question'][:60]}")
        if r.get("matched"):
            report_lines.append(f"    Matched: {', '.join(r['matched'])}")
        if r.get("answer_preview"):
            report_lines.append(f"    Answer:  {r['answer_preview']}...")
        if r.get("error"):
            report_lines.append(f"    Error:   {r['error']}")

    report_lines.append("\n" + "=" * 70)
    report_lines.append(f" SUMMARY: {passed}/{len(results)} passed")
    report_lines.append("=" * 70)

    return "\n".join(report_lines)