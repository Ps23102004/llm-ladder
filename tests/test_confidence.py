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

def test_cosmetic_differences_count_as_agreement():
    # Punctuation, casing and markdown emphasis are not disagreement.
    answers = ["Paris.", "paris", "**Paris**"]
    result, confidence = majority_vote(answers)
    # The original (unnormalized) first-appearing answer is returned.
    assert result == "Paris."
    assert confidence == 1.0

def test_whitespace_and_code_fences_normalize():
    answers = ["`42`", "  42\n", "42"]
    result, confidence = majority_vote(answers)
    assert result == "`42`"
    assert confidence == 1.0

def test_substantively_different_answers_still_disagree():
    answers = ["**Paris**", "London.", "Paris"]
    result, confidence = majority_vote(answers)
    assert result == "**Paris**"
    assert abs(confidence - (2/3)) < 1e-9

def test_hash_suffix_is_not_stripped_as_markdown():
    # "C#" and "F#" must NOT be treated as agreeing with "C"/"F" just because
    # markdown-emphasis stripping used to eat mid-string '#' characters.
    answers = ["C#", "C"]
    result, confidence = majority_vote(answers)
    assert confidence == 0.5

def test_leading_sign_is_significant():
    # "-5" and "5" are different numeric answers; a leading sign must not be
    # stripped as edge punctuation.
    answers = ["-5", "5"]
    result, confidence = majority_vote(answers)
    assert confidence == 0.5

def test_punctuation_only_answers_do_not_degenerate_collapse():
    # "..." and "?" both fully strip to "" under naive trailing-punct removal;
    # normalization must fall back to the original so they don't falsely agree.
    answers = ["...", "?"]
    result, confidence = majority_vote(answers)
    assert confidence == 0.5

def test_snake_case_is_not_stripped():
    answers = ["snake_case", "snakecase"]
    result, confidence = majority_vote(answers)
    assert confidence == 0.5
