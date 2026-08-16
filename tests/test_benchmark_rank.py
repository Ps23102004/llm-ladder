from llm_ladder.benchmark_rank import rank_models


def test_top_scorer_gets_rank_n():
    ranked = rank_models({"a": 0.9, "b": 0.5, "c": 0.7})
    by_model = {r.model: r.rank for r in ranked}
    assert by_model["a"] == 3
    assert by_model["c"] == 2
    assert by_model["b"] == 1

def test_empty_scores_returns_empty_list():
    assert rank_models({}) == []

def test_single_model_gets_rank_one():
    ranked = rank_models({"solo": 0.5})
    assert ranked[0].rank == 1

def test_ties_broken_by_input_order():
    ranked = rank_models({"a": 0.5, "b": 0.5})
    by_model = {r.model: r.rank for r in ranked}
    assert by_model["a"] == 2
    assert by_model["b"] == 1
