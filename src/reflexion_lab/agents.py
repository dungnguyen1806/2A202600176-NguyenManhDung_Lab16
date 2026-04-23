from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .llm_runtime import (
    FAILURE_MODE_BY_QID,
    actor_answer,
    evaluator,
    reflector,
    reset_cumulative_tokens,
    get_last_token_count,
    get_last_latency_ms,
)
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            reset_cumulative_tokens()
            attempt_latency = 0
            
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            attempt_latency += get_last_latency_ms()
            
            judge = evaluator(example, answer)
            attempt_latency += get_last_latency_ms()
            
            token_estimate = get_last_token_count()
            latency_ms = attempt_latency
            
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break
            
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection_entry = reflector(example, attempt_id, judge)
                trace.latency_ms += get_last_latency_ms()
                trace.token_estimate = get_last_token_count()
                
                reflection_memory.append(reflection_entry.next_strategy)
                reflections.append(reflection_entry)
                trace.reflection = reflection_entry
                
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        # Auto-detect failure mode from traces
        if final_score == 1:
            failure_mode = "none"
        else:
            failure_mode = self._detect_failure_mode(traces)
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

    @staticmethod
    def _detect_failure_mode(traces: list[AttemptTrace]) -> str:
        """Classify failure mode based on trace content."""
        reasons = " ".join(t.reason.lower() for t in traces)
        ref_reasons = " ".join(
            (t.reflection.failure_reason.lower() if t.reflection else "")
            for t in traces
        )
        all_text = reasons + " " + ref_reasons

        if "hop" in all_text or "incomplete" in all_text or "first" in all_text:
            return "incomplete_multi_hop"
        if "entity" in all_text or "drift" in all_text or "confused" in all_text:
            return "entity_drift"
        if len(traces) >= 3:
            # All attempts failed — likely reflection didn't help
            return "reflection_overfit"
        return "wrong_final_answer"

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
