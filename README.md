# Legal HSE Retrieval

Проект для retrieval-задачи по корпусу судебных актов НПФ. Целевая метрика - `Recall@5`: для каждого вопроса нужно вернуть до 5 `doc_id`, среди которых должен быть один правильный документ.

Архитектура повторяет план из `deep-research-report.md`: document и passage индексы, sparse retrievers, optional dense retriever, RRF fusion, grouped-validation, метрики и единый путь к `submission.csv`.

## Структура

- `src/legal_hse/` - переиспользуемый Python-пакет.
- `scripts/run_experiments.py` - запуск экспериментов и запись метрик.
- `scripts/run_rerank_experiments.py` - запуск cross-encoder rerank экспериментов.
- `scripts/make_submission.py` - генерация финального сабмита тем же experiment config.
- `main_colab.ipynb` - Colab-файл только для запуска: setup, experiments, push metrics, submission.
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

Для отдельной проверки BGE-M3 candidate retrieval:

```bash
python -m pip install -e ".[dense]"
python scripts/run_experiments.py \
  --data-dir . \
  --mode cv \
  --include-bge-m3 \
  --eval-depth 100 \
  --comparison-baseline rrf_sparse_deep_legal_lemma_char \
  --experiment rrf_sparse_deep_legal_lemma_char \
  --experiment dense_bge_m3_chunk_line_10_5_rd600 \
  --experiment bge_m3_dense_sparse_chunk_line_10_5_rd600 \
  --experiment rrf_sparse_bge_m3_native_line
```

Для cross-encoder rerank поверх recall-oriented кандидатов:

```bash
python -m pip install -e ".[dense]"
python scripts/run_rerank_experiments.py \
  --data-dir . \
  --candidate-experiment rrf_sparse_deep_legal_lemma_char \
  --candidate-experiment quota_sparse_legal_lemma_char_q10
```

По умолчанию rerank использует быстрый screening preset: `mode=holdout`, `depth=20`,
`chunks_per_doc=2`, `chunk_agg=top2_mean`, `score_mode=ce_plus_candidate`, без submission.
Reranker model: `BAAI/bge-reranker-v2-m3`.
Для контрольного сравнения со старым lightweight reranker:

```bash
python scripts/run_rerank_experiments.py \
  --data-dir . \
  --model-name BAAI/bge-reranker-v2-m3 \
  --model-name cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --candidate-experiment rrf_sparse_deep_legal_lemma_char
```

Targeted sweep до двух submission-файлов для free Colab GPU:

```bash
python scripts/run_rerank_experiments.py \
  --data-dir . \
  --candidate-experiment rrf_sparse_deep_legal_lemma_char \
  --depths 20,30 \
  --chunks-per-doc 2,3 \
  --chunk-aggs top2_mean,max_plus_second \
  --score-modes ce_plus_candidate \
  --candidate-score-weights 0.10,0.20,0.30,0.40 \
  --create-submission \
  --submission-top-n 2
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

Отдельные BGE-M3 эксперименты доступны через `recall_candidate_experiments(include_bge_m3=True)`:

- `dense_bge_m3_chunk_line_10_5_rd600` - BGE-M3 dense chunk retrieval через `sentence-transformers`.
- `bge_m3_dense_sparse_chunk_line_10_5_rd600` - native `FlagEmbedding` BGE-M3 dense+sparse chunk retrieval.
- `rrf_sparse_bge_m3_native_line` - RRF sparse baseline branches + native BGE-M3 dense+sparse.
- `quota_sparse_bge_m3_native_line_q8` - recall-oriented quota/union fusion с native BGE-M3.

## Валидация

Для model selection используйте grouped split по `gold_doc_id`, чтобы не завышать качество вопросами про уже знакомые документы.
Процедура соответствует `deep-research-report.md`: `cv` сначала откладывает frozen holdout и гоняет GroupKFold только на оставшемся train-pool; `holdout` отдельно оценивает выбранные конфигурации на той же frozen holdout-части.

```bash
python scripts/run_experiments.py --data-dir . --mode cv
python scripts/run_experiments.py --data-dir . --mode holdout --experiment rrf_bm25_doc_chunk
```

Основные метрики пишутся в `reports/metrics/*.jsonl`, `reports/folds_<run_id>.csv` и `reports/summary_latest.csv`.

`folds_*` хранит raw-строки `fold × experiment × eval_part`: для `mode=cv` это `valid`, для `mode=holdout` это `holdout`, для `mode=train` это `train`. `query_hits_*` хранит per-query hit/miss для paired-сравнений. `summary_*` хранит одну строку на experiment: `mean/std/se` по split'ам, `micro` по объединенным query, а также `delta/wins/losses/ties` против `bm25_doc`, если baseline есть в прогоне. Для одиночного holdout std/se не считаются. `recall@5` остается главной метрикой для выбора submission, остальные recall нужны для диагностики candidate generation и reranking depth.

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
