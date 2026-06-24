from legal_hse.fusion import quota_union_fusion
from legal_hse.retrievers.base import SearchResult


def _result(doc_id: str, score: float, source: str) -> SearchResult:
    return SearchResult(doc_id=doc_id, score=score, source=source, unit_id=doc_id, text=doc_id)


def test_quota_union_fusion_reserves_candidates_from_each_branch():
    first = [_result("a", 10, "first"), _result("b", 9, "first"), _result("c", 8, "first")]
    second = [_result("a", 1, "second"), _result("x", 0.9, "second"), _result("y", 0.8, "second")]

    fused = quota_union_fusion([first, second], quota=2, top_k=4, source="quota")

    assert [item.doc_id for item in fused] == ["a", "x", "b", "y"]
    assert all(item.source == "quota" for item in fused)
