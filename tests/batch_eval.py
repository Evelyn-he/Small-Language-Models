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


def find_first(pattern: str, text: str, cast=float, default=None):
    """Helper: return first regex group or default."""
    m = re.search(pattern, text)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except Exception:
        return default


def run_batch(questions_path: str, user_id: int, output_path: str):
    # Mimic `--verbose` and `--generate_data` flags
    args = SimpleNamespace(verbose=True, generate_data=False)

    # Create Mongo-backed session once (same as in `main_loop`)
    retrievers, router = create_user_session(args, user_id)

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions from {questions_path}")

    results = []

    for idx, question in enumerate(questions, start=1):
        print(f"Running question {idx}/{len(questions)}: {question}")

        # Fresh conversations so each question is independent
        conversation = []
        filtered_convo = []

        # Capture all stdout (debug logs + AI response) from process_message
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

        # Parse metrics from the captured log
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

        # Threshold checks (0.25)
        threshold = 0.25
        slm_conf_above_threshold = (
            slm_conf_parsed is not None and slm_conf_parsed > threshold
        )
        rouge_conf_above_threshold = (
            rouge_confidence is not None and rouge_confidence > threshold
        )

        # Heuristic: did we fall back to the LLM?
        used_llm = llm_response_time is not None

        results.append(
            {
                "user_id": user_id,
                "question": question,
                "answer": reply,
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

    # Write as CSV so we don't require extra Excel dependencies.
    # You can open the CSV directly in Excel.
    df.to_csv(output_path, index=False)

    print(f"Saved results for {len(results)} questions to {output_path}")


if __name__ == "__main__":
    # Paths are relative to this `tests` folder
    base_dir = Path(__file__).resolve().parent
    QUESTIONS_PATH = base_dir / "questions.txt"
    OUTPUT_PATH = base_dir / "batch_results.csv"
    USER_ID = 17850

    run_batch(QUESTIONS_PATH, USER_ID, OUTPUT_PATH)