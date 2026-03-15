"""
Batch evaluation script that runs questions through the pipeline (same as batch_eval.py)
and adds columns comparing each generated answer to reference answers in question_answers.txt.

Input files:
- questions.txt: one question per line.
- question_answers.txt: reference answers (by index or question\\tanswer pairs).
- question_answers_req_span: one line per question (same order as questions). Each line lists
  required spans. Use comma for AND (all must be satisfied) and | for OR (at least one in that
  group). E.g. "yes", "02/13/2010", "yes, 02/13/2010", or "yes|no, 02/13/2010" for (yes OR no) AND date.
  Empty line = no required spans for that question.

Correctness logic:
- When required spans are given for a question: correct iff all required spans are present
  in the generated answer AND cosine similarity passes threshold.
- When no required spans: correct iff cosine similarity passes threshold.
"""

import io
import re
import time
from types import SimpleNamespace
from contextlib import redirect_stdout
from pathlib import Path
import sys

import pandas as pd

# Ensure the project root is on sys.path so `src` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import create_user_session, process_message
from src.context_augmentation.context import embed_text
from sklearn.metrics.pairwise import cosine_similarity


def load_questions(path: str):
    """Load one question per non-empty line from a text file."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if not q or q.startswith("#"):
                continue
            questions.append(q)
    return questions


def load_question_answers(path: str, questions: list[str] | None = None):
    """
    Load reference answers from question_answers.txt.

    Supports two formats:
    1. Answers-only (by index): one reference answer per line, same order as questions.
       Used when no line contains a tab. questions list is required for index alignment.
    2. Explicit Q&A pairs: each line is "question\\tanswer" (tab-separated).
       Returns a dict[question, answer] for lookup. questions list is ignored for lookup.

    Returns:
        Either list[str] (answers by index, length = len(questions)) or
        dict[str, str] (question -> answer). Caller should check type to get reference for each question.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    if not lines:
        return [] if questions is not None else {}

    # If any line has a tab, treat as question\tanswer pairs
    if "\t" in lines[0]:
        qa = {}
        for line in lines:
            if "\t" not in line:
                continue
            q, _, a = line.partition("\t")
            q = q.strip()
            a = a.strip()
            if q:
                qa[q] = a
        return qa

    # Answers-only: one answer per line, same order as questions
    return lines


def load_required_spans(path: str, n_questions: int) -> list[list[list[str]]]:
    """
    Load required spans from question_answers_req_span file.

    One line per question (same order as questions). Each line may be empty (no required
    spans) or use:
    - Comma (,) for AND: each comma-separated part must be satisfied.
    - Pipe (|) for OR within a part: at least one of the | alternatives must be present.
    E.g. "yes|no, 02/13/2010" -> (yes OR no) AND 02/13/2010.

    Returns list of length n_questions. Each element is a list of "OR groups" (each group
    is a list of span strings); all groups must be satisfied (AND).
    """
    result: list[list[list[str]]] = [[] for _ in range(n_questions)]
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n_questions:
                    break
                raw = line.strip()
                if not raw:
                    result[i] = []
                else:
                    # Comma = AND: each segment is one requirement
                    and_groups = [seg.strip() for seg in raw.split(",") if seg.strip()]
                    result[i] = []
                    for seg in and_groups:
                        # Pipe = OR: at least one of these spans must be present
                        or_spans = [s.strip() for s in seg.split("|") if s.strip()]
                        if or_spans:
                            result[i].append(or_spans)
    except FileNotFoundError:
        pass  # all entries remain []
    return result


def _span_in_answer(span: str, generated_answer: str) -> bool:
    """True if span appears in generated answer (date exact, others case-insensitive)."""
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", span):
        return span in generated_answer
    gen_lower = generated_answer.lower()
    span_lower = span.lower()
    if span_lower in gen_lower:
        return True
    # For numeric-looking spans (e.g. 3.39), also accept comma as decimal (3,39)
    if re.search(r"^\d+[.,]\d+$", span.strip()):
        alt = span_lower.replace(".", ",") if "." in span_lower else span_lower.replace(",", ".")
        if alt in gen_lower:
            return True
    return False


def all_required_spans_present(
    required_spans: list[list[str]], generated_answer: str
) -> bool:
    """
    required_spans is a list of OR-groups (each group is list of span strings).
    Return True iff for every OR-group, at least one span in the group is present in the answer.
    """
    if not required_spans:
        return True
    for or_group in required_spans:
        if not any(_span_in_answer(span, generated_answer) for span in or_group):
            return False
    return True


def compute_cosine_similarity(ref: str, gen: str):
    """Cosine similarity in [0, 1] between reference and generated text (embedding)."""
    if not ref or not gen:
        return None
    ref_emb = embed_text(ref).reshape(1, -1)
    gen_emb = embed_text(gen).reshape(1, -1)
    sim = cosine_similarity(ref_emb, gen_emb)[0, 0]
    # clip to [0, 1] in case of numerical noise
    return max(0.0, min(1.0, float(sim)))


def is_correct(
    required_spans: list[list[str]],
    generated_answer: str,
    cosine_sim: float | None,
    cosine_threshold: float,
) -> bool:
    """
    Correct iff at least one of: required spans pass, or cosine similarity passes.
    - When there are required spans: correct if (all spans present) OR (cosine >= threshold).
    - When there are no required spans: correct if cosine >= threshold.
    """
    has_required_spans = len(required_spans) > 0
    spans_ok = (
        all_required_spans_present(required_spans, generated_answer)
        if has_required_spans
        else False
    )
    cosine_ok = cosine_sim is not None and cosine_sim >= cosine_threshold

    if has_required_spans:
        return spans_ok or cosine_ok
    return cosine_ok


def find_first(pattern: str, text: str, cast=float, default=None):
    """Helper: return first regex group or default."""
    m = re.search(pattern, text)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except Exception:
        return default


def run_batch_with_qa(
    questions_path: str,
    question_answers_path: str,
    question_answers_req_span_path: str,
    user_id: int,
    output_path: str,
    cosine_threshold: float = 0.75,
):
    args = SimpleNamespace(verbose=True, generate_data=False)
    retrievers, router = create_user_session(args, user_id)

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    qa_data = load_question_answers(question_answers_path, questions)
    if isinstance(qa_data, dict):
        print(f"Loaded {len(qa_data)} question->answer pairs from {question_answers_path}")
    else:
        print(f"Loaded {len(qa_data)} reference answers (by index) from {question_answers_path}")

    required_spans_by_index: list[list[list[str]]] = load_required_spans(
        question_answers_req_span_path, len(questions)
    )
    n_with_spans = sum(1 for s in required_spans_by_index if s)
    print(f"Loaded required spans for {len(questions)} questions ({n_with_spans} with non-empty spans) from {question_answers_req_span_path}")

    results = []

    for idx, question in enumerate(questions, start=1):
        print(f"Running question {idx}/{len(questions)}: {question}")

        conversation = []
        filtered_convo = []

        buf = io.StringIO()
        start_total = time.time()
        with redirect_stdout(buf):
            reply = process_message(
                user_id=user_id,
                user_input=question,
                args=args,
                conversation=conversation,
                filtered_convo=filtered_convo,
                retrievers=retrievers,
                router=router,
            )
        total_time = time.time() - start_total
        log = buf.getvalue()

        # Reference answer and required spans (from question_answers_req_span file)
        if isinstance(qa_data, dict):
            reference_answer = qa_data.get(question, "")
        else:
            reference_answer = qa_data[idx - 1] if idx - 1 < len(qa_data) else ""

        required_spans = (
            required_spans_by_index[idx - 1]
            if idx - 1 < len(required_spans_by_index)
            else []
        )
        spans_present = all_required_spans_present(required_spans, reply)
        cosine_sim = compute_cosine_similarity(reference_answer, reply) if reference_answer else None
        passes_cosine = (
            (cosine_sim is not None and cosine_sim >= cosine_threshold) if cosine_sim is not None else False
        )
        correct = is_correct(required_spans, reply, cosine_sim, cosine_threshold)

        # Parse metrics from the captured log (same as batch_eval.py)
        slm_conf_raw = find_first(
            r"SLM confidence response:\s*([^\n\r]+)", log, cast=str, default="1"
        )
        slm_conf_parsed = find_first(
            r"Parsed confidence score:\s*([0-9.\-eE]+)", log, cast=float, default=1.0
        )
        rouge_confidence = find_first(
            r"Rouge Confidence Score:\s*([0-9.\-eE]+)", log, cast=float, default=None
        )
        slm_response_time = find_first(
            r"SLM response time:\s*([0-9.\-eE]+)", log, cast=float, default=None
        )
        conf_eval_time = find_first(
            r"Confidence evaluation time:\s*([0-9.\-eE]+)", log, cast=float, default=None
        )
        llm_response_time = find_first(
            r"LLM response time:\s*([0-9.\-eE]+)", log, cast=float, default=None
        )
        routing_time = find_first(
            r"Routing Time:\s*([0-9.\-eE]+)", log, cast=float, default=None
        )
        context_time = find_first(
            r"Context Retrieval Time:\s*([0-9.\-eE]+)", log, cast=float, default=None
        )

        threshold = 0.25
        slm_conf_above_threshold = (
            slm_conf_parsed is not None and slm_conf_parsed > threshold
        )
        rouge_conf_above_threshold = (
            rouge_confidence is not None and rouge_confidence > threshold
        )
        used_llm = llm_response_time is not None

        results.append(
            {
                "user_id": user_id,
                "question": question,
                "answer": reply,
                "reference_answer": reference_answer,
                "required_spans": ", ".join("|".join(or_group) for or_group in required_spans)
                if required_spans
                else "",
                "all_required_spans_present": spans_present,
                "cosine_similarity": cosine_sim,
                "passes_cosine_threshold": passes_cosine,
                "correct": correct,
                "total_time_sec": total_time,
                "routing_time_sec": routing_time,
                "context_retrieval_time_sec": context_time,
                "slm_response_time_sec": slm_response_time,
                "confidence_eval_time_sec": conf_eval_time,
                "llm_response_time_sec": llm_response_time,
                "used_llm_fallback": used_llm,
                "slm_confidence_raw": slm_conf_raw,
                "slm_confidence_parsed": slm_conf_parsed,
                "rouge_confidence_score": rouge_confidence,
                "slm_confident_confidence_gt_0_25": slm_conf_above_threshold,
                "rouge_confident_confidence_gt_0_25": rouge_conf_above_threshold,
                "full_log": log,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Saved results for {len(results)} questions to {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    run_batch_with_qa(
        questions_path=str(base_dir / "questions.txt"),
        question_answers_path=str(base_dir / "question_answers.txt"),
        question_answers_req_span_path=str(base_dir / "question_answers_req_span.txt"),
        user_id=17850,
        output_path=str(base_dir / "batch_results_with_qa.csv"),
        cosine_threshold=0.75,
    )
