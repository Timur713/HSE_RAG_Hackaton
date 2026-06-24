# Legal HSE Retrieval

Проект для retrieval-задачи по корпусу судебных актов НПФ. Целевая метрика - `Recall@5`: для каждого вопроса нужно вернуть до 5 `doc_id`, среди которых должен быть один правильный документ.

Архитектура повторяет план из `deep-research-report.md`: document и passage индексы, sparse retrievers, optional dense retriever, RRF fusion, grouped-validation, метрики и единый путь к `submission.csv`.

## Структура

- `src/legal_hse/` - переиспользуемый Python-пакет.
- `scripts/run_experiments.py` - запуск экспериментов и запись метрик.
- `scripts/make_submission.py` - генерация финального сабмита тем же experiment config.
- `main_colab.ipynb` - один Colab-файл с 4 блоками: setup, experiments, push metrics, submission.
- `reports/metrics/` - JSONL-метрики для GitHub.
- `submissions/` - локальные submission-файлы.

## Быстрый локальный запуск

```bash
python -m pip install -e .
python scripts/run_experiments.py --data-dir . --mode holdout
python scripts/make_submission.py --data-dir . --experiment rrf_bm25_doc_chunk
```

Для dense retrieval и cross-encoder rerank:

```bash
python -m pip install -e ".[dense]"
python scripts/run_experiments.py --data-dir . --include-optional --experiment dense_e5_chunk_line_10_5
```

## Эксперименты

Эксперименты задаются в `legal_hse.experiments.default_experiments()`.

Текущие быстрые варианты:

- `tfidf_word_doc` - TF-IDF по полным документам.
- `tfidf_char_doc_3_5` - char n-gram TF-IDF.
- `bm25_doc` - BM25 по полным документам.
- `bm25_field_aware_doc` - BM25 по документам с извлеченными структурными legal-полями.
- `bm25_chunk_line_10_5_max` - BM25 по line-window чанкам.
- `bm25_chunk_line_8_4_top2` - альтернативный chunk sweep.
- `rrf_bm25_doc_chunk` - RRF по BM25 document + BM25 chunk.
- `rrf_sparse_doc_chunk_char` - RRF по BM25 document + BM25 chunk + char TF-IDF.
- `rrf_sparse_doc_chunk_char_field` - та же fusion-ветка плюс field-aware BM25.
- `dense_e5_chunk_line_10_5` - optional dense E5 chunk retriever.
- `rrf_bm25_dense` - optional sparse+dense hybrid.

## Валидация

Для model selection используйте grouped split по `gold_doc_id`, чтобы не завышать качество вопросами про уже знакомые документы.

```bash
python scripts/run_experiments.py --data-dir . --mode cv
```

Основные метрики пишутся в `reports/metrics/*.jsonl` и `reports/summary_latest.csv`: `recall@5`, `recall@10`, `recall@20`, `recall@50`, время запуска, параметры и статус. `recall@5` остается главной метрикой для выбора submission, остальные recall нужны для диагностики candidate generation и reranking depth.

## Evidence supervision

`gold_evidence_text` используется как источник positive passage для будущего fine-tuning bi-encoder/cross-encoder моделей:

```bash
python scripts/build_supervision.py --data-dir . --window-chars 1200
```

Результат пишется в `artifacts/evidence_pairs.csv` и содержит пары `question -> positive_text` вокруг evidence span.

## Colab workflow

Планируемый сценарий:

1. Вручную загрузить только `main_colab.ipynb`.
2. В первом блоке указать URL публичного GitHub-репозитория и установить проект.
3. Во втором блоке запустить все выбранные эксперименты.
4. В третьем блоке ввести `GITHUB_USERNAME`, `GIT_EMAIL`, `SSH_PRIVATE_KEY_B64` и отправить метрики в GitHub.
5. В четвертом блоке сформировать submission-файл.

CSV-файлы должны лежать в корне репозитория рядом с `train.csv`, `test.csv`, `documents.csv`, `sample_submission.csv`.
