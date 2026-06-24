# Tokenization and Lemmatization Decision - 2026-06-24

Source metrics: `reports/summary_20260624T175149Z.csv`.

## Decision

Use `rrf_sparse_legal_lemma_char` as the current best sparse retrieval configuration for submission experiments.

```python
FINAL_EXPERIMENT = "rrf_sparse_legal_lemma_char"
```

This configuration fuses:

- `bm25_legal_lemma_doc`
- `bm25_legal_lemma_chunk_line_10_5_max`
- `tfidf_char_doc_3_5`

with RRF (`rrf_k=60`, `rank_depth=100`).

## Result Summary

| Experiment | valid Recall@5 | Recall@10 | Recall@20 | Recall@50 | Delta vs `bm25_doc` |
|---|---:|---:|---:|---:|---:|
| `rrf_sparse_legal_lemma_char` | 0.4824 | 0.6364 | 0.7514 | 0.8887 | +0.0501 |
| `rrf_legal_lemma_doc_chunk` | 0.4788 | 0.6179 | 0.7347 | 0.8869 | +0.0464 |
| `bm25_legal_phrase_doc` | 0.4732 | 0.6216 | 0.7496 | 0.8794 | +0.0408 |
| `bm25_legal_lemma_chunk_line_10_5_max` | 0.4732 | 0.5900 | 0.7143 | 0.8739 | +0.0408 |
| `rrf_sparse_doc_chunk_char` | 0.4415 | 0.5881 | 0.7162 | 0.8387 | +0.0093 |
| `bm25_doc` | 0.4322 | 0.5658 | 0.6828 | 0.8257 | baseline |

`rrf_sparse_legal_lemma_char` also has the best `valid_recall@10_mean`, `valid_recall@20_mean`, and `valid_recall@50_mean` among the tested configurations.

## Interpretation

The useful signal is not generic TF-IDF lemmatization by itself. Standalone `tfidf_word_lemma_doc`, `tfidf_word_legal_lemma_doc`, and `tfidf_char_doc_3_5` are all weaker than `bm25_doc`.

The gain comes from legal-aware BM25 normalization:

- Russian lemmatization with `pymorphy3`.
- Preserving legal references such as `ст. 333.19` and `75-ФЗ`.
- Preserving short legal abbreviations such as `НК`, `ГК`, `РФ`, `ОПС`, `ПФР`.
- Removing high-frequency anonymization/noise tokens such as `ФИО` and `адрес`.
- Combining full-document and line-window chunk retrieval.

The char TF-IDF branch is not strong alone, but it adds enough diversity in RRF to make `rrf_sparse_legal_lemma_char` slightly better than the simpler `rrf_legal_lemma_doc_chunk`.

## Residual Risk

The gap between `rrf_sparse_legal_lemma_char` and `rrf_legal_lemma_doc_chunk` is small relative to fold variance. If leaderboard behavior penalizes the char branch, `rrf_legal_lemma_doc_chunk` is the conservative fallback.

For now, keep both in the experiment suite, but use `rrf_sparse_legal_lemma_char` as the default candidate for submission.
