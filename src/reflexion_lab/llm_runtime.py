from __future__ import annotations

import json
import os
import time
import re

from dotenv import load_dotenv
from openai import OpenAI

from .schemas import QAExample, JudgeResult, ReflectionEntry
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .utils import normalize_answer

load_dotenv()

# ---------------------------------------------------------------------------
# OpenAI client setup
# ---------------------------------------------------------------------------
_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MODEL = "deepseek-chat"

# ---------------------------------------------------------------------------
# Token / latency tracking (per-call)
# ---------------------------------------------------------------------------
_last_token_count: int = 0
_last_latency_ms: int = 0
_cumulative_tokens: int = 0


def get_last_token_count() -> int:
    return _cumulative_tokens


def get_last_latency_ms() -> int:
    return _last_latency_ms


def reset_cumulative_tokens() -> None:
    global _cumulative_tokens
    _cumulative_tokens = 0


# ---------------------------------------------------------------------------
# Core LLM call — no retry needed, OpenAI handles rate limits well
# ---------------------------------------------------------------------------
def _call_llm(system: str, user: str) -> str:
    """Call OpenAI API."""
    global _last_token_count, _last_latency_ms, _cumulative_tokens

    start = time.time()
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    _last_latency_ms = int((time.time() - start) * 1000)

    # Token tracking
    if response.usage:
        _last_token_count = response.usage.total_tokens
    else:
        _last_token_count = 0
    _cumulative_tokens += _last_token_count

    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise json.JSONDecodeError("No JSON found", text, 0)


# ---------------------------------------------------------------------------
# Failure mode map (empty — LLM doesn't have fixed mapping)
# ---------------------------------------------------------------------------
FAILURE_MODE_BY_QID: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------
def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> str:
    """Generate an answer using OpenAI, incorporating reflection memory."""
    print(f"  [{agent_type}] qid={example.qid} attempt={attempt_id}", end=" ", flush=True)
    context_str = "\n\n".join(
        f"[{c.title}]: {c.text}" for c in example.context
    )

    memory_str = ""
    if reflection_memory:
        memory_str = (
            "\n\nLessons from previous failed attempts — USE these to avoid repeating mistakes:\n"
            + "\n".join(f"- {m}" for m in reflection_memory)
        )

    user_prompt = (
        f"Question: {example.question}\n\n"
        f"Context:\n{context_str}"
        f"{memory_str}\n\n"
        f"Answer:"
    )

    raw = _call_llm(ACTOR_SYSTEM, user_prompt)
    raw = raw.strip().strip('"').strip("'").rstrip(".")
    if raw.lower().startswith("answer:"):
        raw = raw[7:].strip()
    print(f"-> {raw[:60]}", flush=True)
    return raw


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
def evaluator(example: QAExample, answer: str) -> JudgeResult:
    """Evaluate answer correctness using OpenAI as LLM-as-Judge."""
    if normalize_answer(answer) == normalize_answer(example.gold_answer):
        return JudgeResult(
            score=1,
            reason="Answer matches the gold answer after normalization.",
            missing_evidence=[],
            spurious_claims=[],
        )

    user_prompt = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Agent answer: {answer}\n\n"
        f"Return JSON evaluation:"
    )

    raw = _call_llm(EVALUATOR_SYSTEM, user_prompt)

    try:
        data = _extract_json(raw)
        return JudgeResult(
            score=int(data.get("score", 0)),
            reason=data.get("reason", "Unknown"),
            missing_evidence=data.get("missing_evidence", []),
            spurious_claims=data.get("spurious_claims", []),
        )
    except (json.JSONDecodeError, Exception):
        return JudgeResult(
            score=0,
            reason=f"Evaluator parse error. Raw: {raw[:200]}",
            missing_evidence=[],
            spurious_claims=[answer],
        )


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------
def reflector(
    example: QAExample, attempt_id: int, judge: JudgeResult
) -> ReflectionEntry:
    """Reflect on failure and propose next strategy using OpenAI."""
    user_prompt = (
        f"Question: {example.question}\n"
        f"Wrong answer was evaluated:\n"
        f"- Score: {judge.score}\n"
        f"- Reason: {judge.reason}\n"
        f"- Missing evidence: {judge.missing_evidence}\n\n"
        f"This was attempt #{attempt_id}. Analyze and return JSON:"
    )

    raw = _call_llm(REFLECTOR_SYSTEM, user_prompt)

    try:
        data = _extract_json(raw)
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=data.get("failure_reason", judge.reason),
            lesson=data.get("lesson", "Verify all hops before answering."),
            next_strategy=data.get("next_strategy", "Re-read context carefully."),
        )
    except (json.JSONDecodeError, Exception):
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Need to verify all reasoning hops before answering.",
            next_strategy="Re-read each context paragraph and trace the chain step by step.",
        )