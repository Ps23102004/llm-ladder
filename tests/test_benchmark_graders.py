from llm_ladder.benchmark_graders import (
    grade_reasoning, grade_code, grade_json_extraction, grade_tool_schema,
    grade_instruction_following, grade_factual_recall, grade_rag,
    load_benchmark_tasks, load_rag_corpus,
)


def test_grade_reasoning_correct_number():
    result = grade_reasoning("The speed is 40 mph.", "40")
    assert result.passed

def test_grade_reasoning_wrong_number():
    result = grade_reasoning("The speed is 35 mph.", "40")
    assert not result.passed

def test_grade_reasoning_no_number_fails_cleanly():
    result = grade_reasoning("I don't know.", "40")
    assert not result.passed

def test_grade_code_correct_function():
    answer = "```python\ndef add(a, b):\n    return a + b\n```"
    result = grade_code(answer, "add(2, 3)", "5")
    assert result.passed

def test_grade_code_wrong_output():
    answer = "```python\ndef add(a, b):\n    return a - b\n```"
    result = grade_code(answer, "add(2, 3)", "5")
    assert not result.passed

def test_grade_code_no_code_block_fails_cleanly():
    result = grade_code("I cannot write code.", "add(2, 3)", "5")
    assert not result.passed

def test_grade_code_edge_case_division_by_zero():
    answer = "```python\ndef safe_divide(a, b):\n    return a / b if b != 0 else None\n```"
    result = grade_code(answer, "safe_divide(10, 0)", "None")
    assert result.passed

def test_grade_json_extraction_correct_fields():
    answer = '{"company": "Acme Corp", "role": "Senior Engineer"}'
    result = grade_json_extraction(answer, {"company": "Acme Corp", "role": "Senior Engineer"})
    assert result.passed

def test_grade_json_extraction_missing_field():
    answer = '{"company": "Acme Corp"}'
    result = grade_json_extraction(answer, {"company": "Acme Corp", "role": "Senior Engineer"})
    assert not result.passed

def test_grade_json_extraction_no_json_fails_cleanly():
    result = grade_json_extraction("I found no company.", {"company": "Acme Corp"})
    assert not result.passed

def test_grade_tool_schema_correct_call():
    answer = '{"tool": "get_weather", "args": {"city": "Paris"}}'
    result = grade_tool_schema(answer, "get_weather", ["city"])
    assert result.passed

def test_grade_tool_schema_wrong_tool_name():
    answer = '{"tool": "get_time", "args": {"city": "Paris"}}'
    result = grade_tool_schema(answer, "get_weather", ["city"])
    assert not result.passed

def test_grade_tool_schema_missing_args():
    answer = '{"tool": "get_weather", "args": {}}'
    result = grade_tool_schema(answer, "get_weather", ["city"])
    assert not result.passed

def test_grade_instruction_following_matches_pattern():
    result = grade_instruction_following("clear blue sky", r'^(\S+\s+){2}\S+$')
    assert result.passed

def test_grade_instruction_following_wrong_word_count():
    result = grade_instruction_following("blue", r'^(\S+\s+){2}\S+$')
    assert not result.passed

def test_grade_factual_recall_exact_match():
    result = grade_factual_recall("1889", ["1889"])
    assert result.passed

def test_grade_factual_recall_accepts_any_listed_answer():
    result = grade_factual_recall("It was written by Shakespeare.", ["William Shakespeare", "Shakespeare"])
    assert result.passed

def test_grade_factual_recall_wrong_answer():
    result = grade_factual_recall("1900", ["1889"])
    assert not result.passed

def test_grade_rag_has_required_keywords():
    result = grade_rag("It uses self-consistency voting across samples.", ["self-consistency", "voting"])
    assert result.passed

def test_grade_rag_missing_keyword():
    result = grade_rag("It uses some kind of voting.", ["self-consistency", "voting"])
    assert not result.passed

def test_load_benchmark_tasks_has_all_categories():
    tasks = load_benchmark_tasks()
    for category in ["reasoning", "code", "json_extraction", "tool_schema", "instruction_following", "factual_recall"]:
        assert category in tasks
        assert len(tasks[category]) > 0

def test_load_rag_corpus_has_corpus_and_questions():
    rag = load_rag_corpus()
    assert len(rag["corpus"]) > 0
    assert len(rag["questions"]) > 0
