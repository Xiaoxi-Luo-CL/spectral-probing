# Step 1 Plan: Spectral Profile for Zero-shot Cross-lingual Transfer

## Core Argument

The spectral profile P(lang, task) — a learned frequency filter over the position axis of token
representations — captures **task-specific, language-invariant** contextual scale information
in encoder-only models (after word-level pooling). We argue that this profile, learned from
a high-resource source language A, can be transferred directly to a low-resource target language
B to improve zero-shot and low-resource cross-lingual transfer on the same task.

**The key prediction**: Using the source spectral profile as a frequency filter on target-language
word-pooled representations is better than using no filter (raw representations), because the
filter selects the contextual scale that is genuinely informative for the task — a property that
is more stable across languages than the raw representation distribution.

**Scope**: Primary focus on **encoder-only models** (XLM-R). Decoder models (Qwen and Mistral)
are included as a contrast, since preliminary results suggest word-pooled spectral profiles for
parsing/POS are language-invariant in encoders but not in decoders.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| P(A, T) | Spectral profile (learned frequency filter, 512-dim) trained on language A, task T |
| W(A, T) | Linear probe weights trained on language A, task T |
| `P(A)+W(A) → B` | Apply filter P(A,T) and probe W(A,T) to word-pooled representations of language B |
| `W(A) → B` | Apply probe W(A,T) to raw word-pooled representations of B (no DCT filtering) |

---

## Models

| Model | Type | Role |
|-------|------|------|
| `FacebookAI/xlm-roberta-base` | Encoder-only | **Primary** — task-invariant profiles expected |
| `Qwen/Qwen2.5-7B` | Decoder-only | **Contrast** — profiles expected to vary across languages |
| `mistralai/Mistral-7B-v0.3` | Decoder-only | **Contrast** — second decoder representative |

**Layer**: Final transformer layer for all models. Earlier layers to be considered in future work.

**Word pooling**: Average subword tokens to word-level representations (`pool_word_embed='avg'`).
This controls for tokenization fertility differences across languages, making spectral profiles
comparable. The pooling logic in `encoders.py` (lines 166–171) applies identically to encoder
and decoder models.

---

## Tasks

| Task | Description | Data format |
|------|-------------|-------------|
| `relations` | Dependency head relation classification (UD UPOS head labels) | Token-level, word-pooled |
| `pos` | Universal POS tag classification | Token-level, word-pooled |

Both tasks use Universal Dependencies (UD) treebanks and word-level labels.

---

## Languages and Datasets

### Existing data (already preprocessed)

| Language | Treebank | Train size | Test size | Notes |
|----------|----------|-----------|----------|-------|
| en | UD_English-EWT | 12,545 | 2,078 | Primary source language |
| de | UD_German-GSD | 13,815 | 978 | Germanic |
| fr | UD_French-GSD | 14,451 | 417 | Romance |
| hi | UD_Hindi-HDTB | 13,307 | 1,685 | SOV, Indic script |
| tr | UD_Turkish-IMST | 15,399 | 1,644 | Agglutinative, SOV |
| ja | UD_Japanese-GSD | 7,051 | 544 | SOV, CJK |
| ko | UD_Korean-GSD | 23,011 | 2,288 | SOV, agglutinative |
| zh | UD_Chinese-GSD | 3,998 | 501 | SVO, CJK |

### New data to collect

| Language | Treebank | Estimated size | Notes |
|----------|----------|---------------|-------|
| yue (Cantonese) | UD_Cantonese-HK | ~1,004 total | Download and preprocess with `convert_parsing.py` |

**Note on yue**: With ~1,004 sentences total, few-shot experiments (n=500–2000) are NOT feasible.
Yue is used for zero-shot evaluation only (conditions a and b). Few-shot experiments (condition c)
use zh (3,998 train) as the primary low-resource target.

---

## Experiments

All four conditions are evaluated on the **same test set** of the target language B.

| Condition | Filter on B | Probe W on B | Training on B | Notes |
|-----------|------------|--------------|---------------|-------|
| **(a) Spectral transfer** | P(A, T) frozen | W(A, T) | None | Proposed zero-shot method |
| **(b) No-filter baseline** | None (raw reps) | W(A, T) | None | Standard zero-shot transfer |
| **(c) Few-shot full** | Random init → trained | Trained from scratch | n ∈ {500, 1000, 2000} | Full few-shot upper-reference |
| **(d) Few-shot frozen-P** | P(A, T) frozen | Trained from scratch | n ∈ {500, 1000, 2000} | Key ablation: filter transfers, probe adapts |

**Key comparisons**:
- (a) vs. (b): does P(A) help zero-shot transfer?
- (d) vs. (c): does the source spectral filter P(A) help few-shot probe training vs. random P?
- (d) vs. (a): at what n does adapting W to B data surpass the zero-shot result?

**Source languages for (a), (b), (d)**: en (primary), zh (for zh → yue comparison).

**Target languages**:
- Zero-shot (a, b): all 8 languages + yue
- Few-shot (c, d): zh (primary), plus hi, de, tr as secondary targets (sufficient training data)

**Primary pairs of interest**: en → zh, zh → yue, en → yue.

---

## Evaluation Metrics

| Task | Metric |
|------|--------|
| `relations` | Accuracy (head relation classification, word-level) |
| `pos` | Accuracy (UPOS classification, word-level) |

---

## Data and Compute Requirements

### Results index

A full index of all 450 trained probes (with `best.pt`) is at:
`research_plan/results_index.json`

Lookup key format: `model|language|task|pool_word_embed`
Example: `FacebookAI/xlm-roberta-base|en|relations|avg`

Each `best.pt` contains: `frequency_filter` (P, 512-dim), `classifier` (W, Linear),
`classes` (label list), `language_model_name`.

### Availability for Step 1

| Model | Task | Pool | Languages available | Status |
|-------|------|------|---------------------|--------|
| xlm-roberta-base | relations | avg | de, en, hi, ja, ko, tr, zh | **Ready** ✓ |
| xlm-roberta-base | pos | avg | de, en, ja, zh | **Partial** — missing hi, ko, tr |
| xlm-roberta-base | pos | last | de, en, hi, ja, ko, tr, zh | **Ready** ✓ |
| Qwen2.5-7B | — | — | — | **Not run yet** |
| Mistral-7B-v0.3 | — | — | — | **Not run yet** |

**Primary setting for conditions (a) and (b)**: `pool=avg`, task=`relations`, XLM-R (full 7-language
coverage). For POS: use `pool=last` (7 langs) or run 3 new experiments (xlm-r + pos + avg + hi/ko/tr).

### New experiments needed

| What | Why |
|------|-----|
| UD_Cantonese-HK download + `convert_parsing.py` | yue data for zero-shot evaluation |
| xlm-r + yue + relations/pos + pool=avg | Oracle P(yue) needed for analysis; source probe for yue |
| (Optional) xlm-r + pos + avg + hi, ko, tr | Complete avg-pooled POS coverage |
| Qwen2.5-7B + all langs + relations/pos + pool=avg | Decoder contrast experiments |
| Mistral-7B-v0.3 + all langs + relations/pos + pool=avg | Decoder contrast experiments |

---

## Implementation

**Script**: `transfer/zero_shot.py` (run from `spectral-probing/` directory)

```bash
# Condition (a) — zero-shot spectral transfer
python transfer/zero_shot.py \
  --source_exp results/<en-relations-exp> \
  --target_test tasks/ud-syntax/zh/zh-gsd-test-relations.csv \
  --condition a --pool_word_embed avg \
  --output_dir transfer/results/en-zh-relations

# Condition (b) — zero-shot baseline (no filter)
python transfer/zero_shot.py \
  --source_exp results/<en-relations-exp> \
  --target_test tasks/ud-syntax/zh/zh-gsd-test-relations.csv \
  --condition b --pool_word_embed avg \
  --output_dir transfer/results/en-zh-relations

# Condition (c) — few-shot, train P+W from scratch
python transfer/zero_shot.py \
  --source_exp results/<en-relations-exp> \
  --target_test tasks/ud-syntax/zh/zh-gsd-test-relations.csv \
  --target_train tasks/ud-syntax/zh/zh-gsd-train-relations.csv \
  --condition c --n_examples 500 --pool_word_embed avg \
  --output_dir transfer/results/en-zh-relations-n500

# Condition (d) — few-shot, freeze P(A), train W only
python transfer/zero_shot.py \
  --source_exp results/<en-relations-exp> \
  --target_test tasks/ud-syntax/zh/zh-gsd-test-relations.csv \
  --target_train tasks/ud-syntax/zh/zh-gsd-train-relations.csv \
  --condition d --n_examples 500 --pool_word_embed avg \
  --output_dir transfer/results/en-zh-relations-n500
```

**Condition details in `zero_shot.py`**:
- **(a)**: builds encoder with `source_P` (frq_tuning=False), uses `source_W` → zero-shot eval
- **(b)**: builds encoder with no filter, uses `source_W` → zero-shot eval
- **(c)**: builds encoder with random P init (frq_tuning=True), fresh W → trains both on n examples
- **(d)**: builds encoder with `source_P` (frq_tuning=False), fresh W → trains only W on n examples

Early stopping for (c)/(d): uses 20% of sampled n examples as dev (or `--target_dev` if provided).
Output: one `result_condX[_nN].json` per run saved in `--output_dir`.

**Execution order**:
1. Verify script on en → de (relations, pool=avg) — both source and target results are in the index
2. Run (a) and (b) for all language pairs (relations first, pool=avg)
3. Run (c) and (d) for en → zh at n ∈ {500, 1000, 2000}
4. Download UD_Cantonese-HK, preprocess, run (a)/(b) for zh → yue and en → yue
5. Decoder contrast: train probes on Qwen2.5-7B and Mistral, repeat (a)/(b)

---

## Open Questions

1. **yue few-shot**: If we want even a small few-shot experiment on yue, limited to n ≤ 700
   (full training set). Consider n ∈ {50, 100, 200} as supplementary for yue only.

2. **Universal profile (optional)**: A P_univ = mean of P(lang, T) across all source languages
   could be added as condition (e) later if (a) vs. (b) results are inconclusive. Not
   included in the initial scope.

3. **POS pool setting decision**: Use `pool=last` for POS (7 langs already available) or run
   3 new `pool=avg` experiments (hi, ko, tr)? Recommend `pool=last` to avoid new runs, but
   note this introduces a pooling inconsistency between tasks.

## E3: Correlation Analysis (deferred)

After running (a) and (b) for all language pairs, compute:
1. Transfer benefit = acc(a) − acc(b) for each (source A, target B) pair
2. Spectral L1 distance = ||P(A, T) − P(B, T)||₁ (using existing word-pooled profiles)
3. Typological distance from lang2vec (URIEL syntax + morphology features)
4. Correlate both predictors with transfer benefit (Spearman ρ)

**Prediction**: spectral distance is a better predictor than typological distance for encoder
models, because it captures model-internal representation structure.


## 4/15 Experiments for CS 698 Report
1. 4 models (mBERT, XLM-R, aya, Qwen) 
2. 4 high-resource language (all supported), 8 mid, 8 low
3. These languages have at least 200 sentences. Use at least 100 sentences for evaluation. For the rest, experiment under conditions (c)-(e) with 50, 100, 200, 500, 1000 samples for training