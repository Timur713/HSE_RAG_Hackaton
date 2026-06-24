import pandas as pd

from legal_hse.submission import make_submission_frame, validate_submission


def test_submission_validation_accepts_top5_format():
    test = pd.DataFrame({"qid": ["q1", "q2"], "question": ["a", "b"]})
    documents = pd.DataFrame({"doc_id": ["d1", "d2", "d3"], "text": ["", "", ""]})
    submission = make_submission_frame(["q1", "q2"], [["d1", "d2"], ["d3"]])
    validate_submission(submission, test, documents)


def test_submission_dedupes_inside_qid():
    frame = make_submission_frame(["q1"], [["d1", "d1", "d2"]])
    assert frame["doc_id"].tolist() == ["d1", "d2"]
