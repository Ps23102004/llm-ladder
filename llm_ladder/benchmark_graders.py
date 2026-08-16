from __future__ import annotations

import importlib.resources
import json
import re
import subprocess
from dataclasses import dataclass

import yaml


@dataclass
class GradeResult:
    passed: bool
    detail: str


def load_benchmark_tasks() -> dict:
    resource = importlib.resources.files("llm_ladder").joinpath("benchmark_tasks.yaml")
    with resource.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rag_corpus() -> dict:
    resource = importlib.resources.files("llm_ladder").joinpath("rag_corpus.yaml")
    with resource.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_code_block(text: str) -> str | None:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    if "def " in text:
        return text
    return None


def grade_reasoning(answer: str, expected: str) -> GradeResult:
    match = re.search(r"-?\d+(\.\d+)?", answer)
    if not match:
        return GradeResult(False, f"no number found in answer: {answer!r}")
    got = match.group(0)
    passed = got.strip() == expected.strip()
    return GradeResult(passed, f"expected {expected}, got {got}")


def grade_code(answer: str, test_input: str, expected_output: str) -> GradeResult:
    code = _extract_code_block(answer)
    if code is None:
        return GradeResult(False, "no code block found in answer")
    script = f"{code}\nprint({test_input})"
    try:
        proc = subprocess.run(["python3", "-c", script], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return GradeResult(False, "code execution timed out")
    if proc.returncode != 0:
        return GradeResult(False, f"code raised an error: {proc.stderr.strip()[:200]}")
    got = proc.stdout.strip()
    passed = got == expected_output.strip()
    return GradeResult(passed, f"expected {expected_output!r}, got {got!r}")


def grade_json_extraction(answer: str, expected_fields: dict) -> GradeResult:
    obj = _extract_json(answer)
    if obj is None:
        return GradeResult(False, "no valid JSON object found in answer")
    missing = [k for k in expected_fields if k not in obj]
    if missing:
        return GradeResult(False, f"missing fields: {missing}")
    mismatched = {
        k: obj[k] for k in expected_fields
        if str(obj[k]).strip().lower() != str(expected_fields[k]).strip().lower()
    }
    if mismatched:
        return GradeResult(False, f"field mismatch: {mismatched}")
    return GradeResult(True, "all fields matched")


def grade_tool_schema(answer: str, expected_tool: str, expected_args_keys: list[str]) -> GradeResult:
    obj = _extract_json(answer)
    if obj is None:
        return GradeResult(False, "no valid JSON tool call found")
    if obj.get("tool") != expected_tool:
        return GradeResult(False, f"expected tool '{expected_tool}', got {obj.get('tool')!r}")
    args = obj.get("args", {})
    if not isinstance(args, dict):
        return GradeResult(False, "'args' is not an object")
    missing = [k for k in expected_args_keys if k not in args]
    if missing:
        return GradeResult(False, f"missing args: {missing}")
    return GradeResult(True, "tool call well-formed")


def grade_instruction_following(answer: str, pattern: str) -> GradeResult:
    passed = re.match(pattern, answer.strip()) is not None
    return GradeResult(passed, f"pattern {pattern!r} {'matched' if passed else 'did not match'} {answer.strip()!r}")


def grade_factual_recall(answer: str, accepted: list[str]) -> GradeResult:
    normalized = answer.strip().lower()
    passed = any(acc.strip().lower() in normalized for acc in accepted)
    return GradeResult(passed, f"expected one of {accepted}, got {answer.strip()!r}")


def grade_rag(answer: str, required_keywords: list[str]) -> GradeResult:
    normalized = answer.lower()
    missing = [kw for kw in required_keywords if kw.lower() not in normalized]
    passed = not missing
    return GradeResult(passed, f"missing keywords: {missing}" if missing else "all keywords present")
