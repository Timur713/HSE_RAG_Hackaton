import numpy as np

from legal_hse.retrievers.bge_m3 import _lexical_weights_to_csr, _top_indices


def test_lexical_weights_to_csr_uses_existing_query_vocabulary():
    token_to_col: dict[str, int] = {}
    passages = _lexical_weights_to_csr(
        [{"1": 0.5, "2": 1.0}, {"2": 0.25, "3": 0.75}],
        token_to_col=token_to_col,
        allow_new=True,
    )
    query = _lexical_weights_to_csr(
        [{"2": 2.0, "4": 100.0}],
        token_to_col=token_to_col,
        allow_new=False,
        n_cols=passages.shape[1],
    )

    scores = (query @ passages.T).toarray()[0]

    assert scores.tolist() == [2.0, 0.5]
    assert "4" not in token_to_col


def test_top_indices_returns_descending_scores():
    scores = np.asarray([0.1, 0.7, 0.3, 0.9], dtype="float32")

    assert _top_indices(scores, 3).tolist() == [3, 1, 2]
