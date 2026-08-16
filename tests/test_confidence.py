import pytest
from llm_ladder.confidence import majority_vote

def test_empty_list_raises_value_error():
    with pytest.raises(ValueError):
        majority_vote([])

def test_all_agree_returns_one():
    answers = ["hello", "hello", "hello"]
    result, confidence = majority_vote(answers)
    assert result == "hello"
    assert confidence == 1.0

def test_no_agreement_returns_first_appearing_fraction():
    # All distinct answers
    answers = ["A", "B", "C"]
    result, confidence = majority_vote(answers)
    # The spec says: "no-agreement among distinct answers returns fraction 1/n for the first-appearing one"
    # First appearing is "A". n=3. Fraction 1/3.
    assert result == "A"
    assert abs(confidence - (1/3)) < 1e-9

def test_exact_tie_returns_first_appearing():
    # Tie between "A" and "B"
    # Input: ["A", "B", "A", "B"] -> Count A:2, Count B:2.
    # First appearing in input is "A".
    answers = ["A", "B", "A", "B"]
    result, confidence = majority_vote(answers)
    assert result == "A"
    # Confidence is 2/4 = 0.5
    assert confidence == 0.5

def test_majority_wins():
    answers = ["A", "A", "B"]
    result, confidence = majority_vote(answers)
    assert result == "A"
    assert confidence == 2/3
