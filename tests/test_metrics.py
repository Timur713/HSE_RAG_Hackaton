from legal_hse.metrics import evaluate_predictions, recall_at_k


def test_recall_at_k_dedupes_and_truncates():
    gold = ["d1", "d2", "d3"]
    preds = [["d0", "d1", "d1"], ["d4", "d5", "d2"], ["d4", "d5", "d6"]]
    assert recall_at_k(gold, preds, k=2) == 1 / 3
    assert recall_at_k(gold, preds, k=3) == 2 / 3


def test_evaluate_predictions_has_expected_keys():
    metrics = evaluate_predictions(["d1"], [["d1"]])
    assert metrics["recall@1"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr@10"] == 1.0
