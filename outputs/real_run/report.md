# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: gemini
- Records: 240
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.6333 | 0.825 | 0.1917 |
| Avg attempts | 1 | 1.5583 | 0.5583 |
| Avg token estimate | 1164.02 | 3306.69 | 2142.67 |
| Avg latency (ms) | 3074.25 | 5566.62 | 2492.37 |

## Failure modes
```json
{
  "react": {
    "none": 76,
    "wrong_final_answer": 32,
    "incomplete_multi_hop": 9,
    "entity_drift": 3
  },
  "reflexion": {
    "none": 99,
    "reflection_overfit": 12,
    "incomplete_multi_hop": 8,
    "entity_drift": 1
  },
  "overall": {
    "none": 175,
    "wrong_final_answer": 32,
    "incomplete_multi_hop": 17,
    "entity_drift": 4,
    "reflection_overfit": 12
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion demonstrates a clear improvement over ReAct on multi-hop QA tasks. The key benefit is the self-correction loop: when the Actor fails on the first attempt (typically by stopping at hop 1 or drifting to a wrong entity), the Evaluator provides structured feedback identifying the specific failure mode. The Reflector then converts this feedback into an actionable lesson stored in episodic memory. On subsequent attempts, the Actor uses these lessons to avoid repeating the same mistake. The primary tradeoff is increased cost and latency — Reflexion uses 2-3x more API calls per question. Failure modes that persist include entity drift (when context contains similar entities) and reflection overfit (when the reflection memory becomes too generic to be useful). Evidence-grounded evaluation is critical: vague evaluator feedback leads to vague reflections, which fail to improve the next attempt.
