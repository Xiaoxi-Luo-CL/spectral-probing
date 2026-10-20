# Replicating the spectral-transfer experiments (Direction 1)

This document explains how to rerun, from scratch, the cross-lingual **spectral transfer** experiments on branch `sync-1020`. It covers four things:

1. The **v3 transfer grid**: 4 models × 2 source languages × 24 target languages × 13 settings × 3 seeds.
2. The **related-language pilot**: a probe trained on 100 target sentences, with the spectral filter taken from a related or an unrelated source language.
3. Two **protocol checks** that are still open: the XLM-R learning rate, and a constant vs a decaying learning-rate schedule.
4. The **analysis** scripts and figures for all of the above.

Directions 2 and 3 (`direction2/`, `direction3/`) are on this branch but are not covered here.

---

## 1. What the experiments do

A **spectral probe** has two parts:

- **γ (gamma), a frequency filter.** The sequence of word embeddings of a sentence is transformed with the DCT along the sequence axis. Each frequency is multiplied by a learned gate, `sigmoid(logit)`, and the result is transformed back.
- **W, a linear probe.** It classifies each word's dependency relation (`relations` task, coarse UD labels) from the filtered embedding.

Embeddings are pooled to words: the average of subword tokens for encoders, the last subword token for decoders.

We first train a probe on a **source** language (en or zh), which gives the source probe **W_A** and the source filter **γ_A**. We then move to a **target** language and vary what is reused. W is the linear probe, γ is the frequency filter:

| W option | meaning |
|---|---|
| `froze` | use W_A as is |
| `tune` | start from W_A and train on the target |
| `scratch` | a new probe trained on the target only |

| γ option | meaning |
|---|---|
| `none` | no filter |
| `froze` | use γ_A as is |
| `tune` | start from γ_A and train on the target |
| `scratch` | a new filter, initialized like the source's, trained on the target |

A **setting** is named `W<w>_G<g>`, for example `Wscratch_Gfroze` (new probe, frozen source filter). There are 12 combinations plus `plain`, 13 in total:

- **Zero-shot settings** use no target data: `Wfroze_Gnone`, `Wfroze_Gfroze` and `plain`.
  - `plain` applies a separate source probe that was **trained without any filter** (W_A⁰). It is the standard zero-shot baseline.
  - `Wfroze_Gnone` applies the filter-trained W_A to unfiltered embeddings.
- **The other 10 settings train on the target.** `Wscratch_Gnone` (new probe, no filter) is the standard few-shot baseline.

**Training data size n.** n counts target sentences for training *plus* development: they are split 80/20, and the dev part is used for early stopping.

**Metric.** Test accuracy over words, counting only words whose gold label is in the probe's label set. Each batch is weighted by the number of words it scored, not by a per-batch average.

### Models

| model (Hugging Face ID) | word pooling | embedding batch | notes |
|---|---|---|---|
| `bert-base-multilingual-cased` | `avg` | 32 | |
| `FacebookAI/xlm-roberta-base` | `avg` | 32 | |
| `CohereLabs/aya-expanse-8b` | `last` | 8 | **gated**: accept the license on Hugging Face and log in |
| `Qwen/Qwen2.5-7B` | `last` | 8 | |

The 7–8B decoders run in fp32 and need a **48 GB GPU** at embedding batch 8. Probe training always uses batch 32: the embeddings are computed once, cached, and the probe trains on the cache.

---

## 2. Environment

The runs used Python 3.10 with these packages:

```
torch==2.7.1 (CUDA 11.8 build)   transformers==4.51.0   tokenizers==0.21.4
numpy==2.2.6   scipy==1.15.3   matplotlib==3.10.3   scikit-learn==1.7.2
pandas==2.3.3   accelerate==1.7.0   sentencepiece==0.2.0   safetensors==0.5.3
protobuf==6.31.1   huggingface_hub==0.36.2
```

```bash
conda create -n spectral python=3.10 -y && conda activate spectral
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118   # pick the CUDA build for your GPUs
pip install transformers==4.51.0 tokenizers==0.21.4 numpy==2.2.6 scipy==1.15.3 matplotlib==3.10.3 \
    scikit-learn==1.7.2 pandas==2.3.3 accelerate==1.7.0 sentencepiece==0.2.0 safetensors==0.5.3 \
    protobuf==6.31.1 huggingface_hub==0.36.2 flake8
```

Run every command from the repository root, `spectral-probing/`.

### Adapting the job scripts to your cluster

The SLURM scripts contain settings specific to our cluster. Edit them before submitting anything:

| what | where | change to |
|---|---|---|
| `PYTHON="/u501/x25luo/.conda/envs/grounding/bin/python"` | `train_source.slurm`, `direction1/scripts/slurm_transfer.sh`, `direction1/scripts/submit_schedule_check.sh` (the inline Python block) | your environment's `python` |
| `#SBATCH --partition=ALL`, `--exclude=...`, `--mem=100GB`, `--time=...` | `train_source.slurm`, `direction1/scripts/slurm_transfer.sh` | your partition and nodes; keep ≥ 48 GB GPU and about 100 GB RAM for the decoders |
| `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` | same two files | keep these only if the four models are already in the HF cache on the compute nodes; otherwise download them first or remove the line |

Job logs go to `slurm/JOB-<id>.out` and `.err`. Create the directory first with `mkdir -p slurm`.

---

## 3. Data

### 3.1 Universal Dependencies v2.17

Download UD v2.17 from <https://universaldependencies.org> and unpack it so that treebanks sit at `data/ud-treebanks-v2.17/UD_<Language>-<Treebank>/` (for example `data/ud-treebanks-v2.17/UD_Czech-CAC/cs_cac-ud-train.conllu`). `data/` is not in git.

### 3.2 Per-language relation files

Each language needs a `tasks/ud-syntax/<lang>/relations/<lang>-<treebank>-{train,dev,test}-relations.csv`. These are the source training data, and also the input for the high-resource targets.

- **Already in git:** en (EWT), zh (GSD), de (GSD), es (GSD), fr (GSD), ja (GSD). Do **not** regenerate these: they come from an earlier UD release than v2.17, and the results use these exact files.
- **Generate from UD v2.17** (checked to be identical to the files we used):

```bash
cd tasks/ud-syntax
for p in "hi UD_Hindi-HDTB" "ko UD_Korean-Kaist" "tr UD_Turkish-Kenet" "vi UD_Vietnamese-VTB" \
         "cs UD_Czech-CAC" "ca UD_Catalan-AnCora" "it UD_Italian-ISDT"; do
    set -- $p
    python convert.py ../../data/ud-treebanks-v2.17/$2 relations $1/relations/
done
cd ../..
```

`convert.py` writes one CSV per split, with `text` (space-separated words) and `label` (one coarse relation per word) columns.

### 3.3 Target files

```bash
python -m direction1.transfer.prepare_coverage_data   # 24 targets of the v3 grid (+ coverage_targets.json)
python -m direction1.transfer.prepare_pair_data       # related-pair targets nds, oc, scn (+ pair_targets.json)
```

Both scripts write merged files `tasks/ud-syntax/coverage-relations/<lang>-all-relations.csv`.

- **`prepare_coverage_data`** merges every split of the low- and mid-resource treebanks, and the three CSVs of the high-resource languages from 3.2.
- **`prepare_pair_data`** keeps the official train/dev/test order for treebanks that have a train split. It refuses to overwrite a file that the v3 grid uses.

### 3.4 Verify the data

Every input file must match the ones we used:

```bash
md5sum -c direction1/transfer/data_checksums.md5     # every line must say OK
```

This covers all 27 target files, the source and high-resource CSVs, `coverage_targets.json` and `pair_targets.json`. If a file fails, the results will not be comparable. Check the UD version, and that the tracked CSVs were not regenerated.

### 3.5 The 24 v3 targets

`n_test` is the fixed test set: the **last** `n_test` sentences of each merged file. The **train pool** is everything before it. A run trains on n sentences sampled from the pool.

| tier | n_test | targets (UD treebanks) |
|---|---|---|
| 200–1000 sentences | 100 | kmr (Kurmanji), hsb (UFAL), sah (YKTDT), bho (BHTB), sq (STAF + TSA), br (KEB), ml (UFAL), mr (UFAL) |
| 1000–2000 sentences | 200 | bm (CRB), quc (IU), yue (HK), ab (AbNC), hu (Szeged), jv (CSUI), ta (MWTT + TTB), kk (KTB) |
| high | 500 | vi (VTB), fr (GSD), de (GSD), es (GSD), tr (Kenet), hi (HDTB), ko (Kaist), ja (GSD) |

---

## 4. Step 1: train the source probes

Every source probe uses the same recipe:
- a linear probe on word-pooled embeddings;
- Adam, lr 1e-3, halving the lr when dev loss plateaus;
- up to 30 epochs, stopping after 3 epochs without improvement;
- batch 32, seed 42.

Filtered probes use `auto(512)`: 512 filter logits, initialized as `ones` + noise 0.1, scale 1.0.

```bash
# usage: sbatch train_source.slurm <model> <lang> <pool> [filter, default 'nofilter()']
sbatch train_source.slurm bert-base-multilingual-cased zh avg 'auto(512)'   # filtered (gives W_A, gamma_A)
sbatch train_source.slurm bert-base-multilingual-cased zh avg               # filter-free (gives W_A^0, for plain)
```

| experiment | languages | filter | models |
|---|---|---|---|
| v3 grid | en, zh | `auto(512)` **and** `nofilter()` | all four (16 jobs) |
| related pairs | hi, cs, de, ca, it, plus en and zh from above | `auto(512)` | mBERT (`avg`), Aya (`last`) (10 more jobs) |

`train_source.slurm` maps each language to its treebank: en `ewt`, zh `gsd`, hi `hdtb`, de `gsd`, cs `cac`, ca `ancora`, it `isdt`. Each run writes `results/<timestamp>_<model>_relations_<lang>/`, which contains `best.pt`, `config.json`, `classify.log` and `spectral_profile.png`.

### Register the source probes

The transfer jobs find source probes through two files:

```bash
# 1. lookup "<model>|<lang>|relations|<pool>" -> filtered run (prefers auto(512) over nofilter())
python direction1/update_results_index.py --results_dir results --index direction1/transfer/results_index.json

# 2. the filter-free runs used by the "plain" setting
python - <<'EOF'
import json
idx = json.load(open('direction1/transfer/results_index.json'))
plain = {f"{e['model']}|{e['language']}|{e['task']}|{e['pool_word_embed']}": e['path']
         for e in idx['entries'] if e['filter'] == 'nofilter()' and e['task'] == 'relations'}
json.dump(plain, open('direction1/transfer/plain_sources.json', 'w'), indent=2)
print(len(plain), 'plain sources')   # 8 for the v3 grid
EOF
```

Train **only one run per (model, language, pool, filter)**. When there are duplicates, the index keeps the most accurate one, which is not what you want. The `results_index.json` in git points to our cluster's paths, so the rebuild above replaces it.

---

## 5. Step 2: the v3 transfer grid

### 5.1 Protocol

| item | value |
|---|---|
| training size n | the largest of {100, 500, 1000} that fits the target's train pool (`--largest_n_only`) |
| split | n sentences sampled from the pool with the run's seed, then split 80/20 into train/dev (same seed → same sample for every setting) |
| optimizer | Adam, initial lr 1e-3, batch 32 |
| learning-rate schedule | `ReduceLROnPlateau`: halve the lr after any epoch whose dev loss does not improve by ≥ 0.1% (relative) |
| stopping | up to 100 epochs; stop after 3 epochs in a row without a ≥ 0.1% relative improvement (`--early_stop 2 --min_delta 1e-3`); restore the best-dev-loss checkpoint |
| seeds | 42, 43, 44; the zero-shot settings train nothing, so they run only in the seed-42 job |
| frozen filters | applied through the sigmoid gate, as in training |
| new (`scratch`) filters | initialized with the source run's recipe, read from its `config.json` |

### 5.2 Submit

```bash
bash direction1/scripts/submit_transfer_v3.sh --dry-run   # 24 jobs: 4 models x {en, zh} x seeds 42-44
bash direction1/scripts/submit_transfer_v3.sh
```

Each job runs one (model, source, seed) over all 24 targets via `direction1/scripts/slurm_transfer.sh` → `python -m direction1.transfer.run_batch`. Before training, it computes the embeddings of every target sentence once and caches them.

**Optional environment overrides**, all documented in the script header:
- `MODELS`, `SEEDS`, `SETTINGS`;
- `OUTPUT_DIR` (default `direction1/transfer/results_coverage_v3`);
- `LR` (initial learning rate).

**Resubmitting is safe:** existing result files are skipped, so a crashed or timed-out job can simply be submitted again.

### 5.3 Output and completeness

Results are written as one JSON per cell:

```
direction1/transfer/results_coverage_v3/<model>/<src>2<tgt>/relations/<setting>[_n<N>]/seed<S>/result_<setting>[_n<N>].json
```

Each JSON holds:
- `test_accuracy` (0–1), `n_examples`, `n_test`;
- `setting`, `source_exp`, `learning_rate`, `random_seed`;
- `test_accuracy_batch_mean`: the old per-batch average, kept for auditing only.

When complete, each model has **624** files for seed 42 (2 sources × 24 targets × 13 settings), and **480** each for seeds 43 and 44 (10 trained settings).

```bash
for m in direction1/transfer/results_coverage_v3/*/; do for s in 42 43 44; do
  echo "$(basename $m) s$s $(find $m -path "*seed$s/*.json" | wc -l)"; done; done
```

---

## 6. Step 3: analysis and figures

```bash
python -m direction1.transfer.analyze_v3                    # all sections, Markdown tables to stdout
python -m direction1.transfer.analyze_v3 --section gamma    # or one section at a time
python -m direction1.transfer.plot_v3 --fmt pdf             # 16 figures -> direction1/transfer/figures_v3/
```

| `analyze_v3` section | question it answers |
|---|---|
| `means` | mean test accuracy of each setting per (model, source), over the 24 targets |
| `paired` | paired setting differences over targets (bootstrap 95% CI, wins) |
| `best` | which setting is best per target; how often a target-only probe wins, with a binomial test against chance |
| `majority` | majority-class accuracy of each test set next to the zero-shot settings |
| `tiers` | filter gains grouped by resource tier, pretraining support, n and encoder/decoder |
| `gamma` | within each W option, each filter form vs no filter (Wilcoxon over targets, Holm-corrected; consistency across the 8 runs) |
| `baselines` | every setting vs `plain` (zero-shot) or `Wscratch_Gnone` (few-shot) |
| `pertarget` | per source and model: in how many of the 24 targets a comparison is significant over seeds |
| `zeroshot` | `Wfroze_Gfroze` vs `plain` per target, with a conservative test over test tokens |
| `select` | the best source-reusing setting vs `Wscratch_Gnone`, picked on test (biased) or on the other targets (unbiased) |
| `bug` | the effect of the early "raw-logit frozen filter" bug: needs the old `results_coverage` (v1) and `results_coverage_v2` grids, which are not part of this replication; skip it |

`majority`, `zeroshot` and `bug` read the source checkpoints through `results_index.json`, so run step 1's registration first.

`plot_v3` draws two figures per (model, source):
- `heatmap_*`: accuracy for every setting and target;
- `delta_*`: each trained setting minus `Wscratch_Gnone`, with a dot where all seeds agree in sign.

### Reference numbers

A complete rerun should reproduce the `means` section closely. The table below shows a few of our values (mean test accuracy in %, over 24 targets). On different GPUs or library builds, expect differences of a few tenths of a point. Bit-identical results need the same hardware and software.

| setting | mBERT en | mBERT zh | XLM-R en | XLM-R zh | Qwen en | Qwen zh | Aya en | Aya zh |
|---|---|---|---|---|---|---|---|---|
| `plain` | 40.7 | 35.8 | 49.6 | 40.9 | 18.6 | 10.5 | 19.1 | 19.5 |
| `Wfroze_Gfroze` | 41.6 | 36.5 | 49.5 | 41.0 | 21.0 | 14.2 | 22.5 | 20.0 |
| `Wtune_Gnone` | 67.3 | 67.2 | 70.5 | 70.1 | 60.4 | 62.1 | 61.5 | 65.3 |
| `Wtune_Gfroze` | 68.1 | 68.2 | 70.5 | 70.2 | 62.2 | 63.0 | 63.9 | 65.6 |
| `Wscratch_Gnone` | 68.3 | 68.5 | 70.1 | 70.5 | 62.2 | 62.2 | 64.7 | 65.0 |
| `Wscratch_Gfroze` | 68.6 | 69.1 | 70.1 | 70.4 | 62.9 | 63.1 | 65.9 | 65.5 |
| `Wscratch_Gtune` | 68.5 | 68.9 | 70.1 | 70.6 | 63.2 | 63.1 | 65.9 | 65.8 |

---

## 7. Step 4: the related-language pilot

**Question.** At a budget of 100 target sentences, does a frozen filter from a **related** high-resource language help more than one from an unrelated source (zh, or en for yue)?

**Pairs** (from `direction1/transfer/related_pairs.json`):

| related source → target | unrelated source used for comparison |
|---|---|
| hi → bho | zh |
| cs → hsb | zh |
| de → nds | zh |
| ca → oc | zh |
| it → scn | zh |
| zh → yue | en |

Every source runs on all 6 targets, so all sources are scored on the same test sets.

### Protocol

These are the `run_batch` flags set by `submit_pairs.sh`:

| item | value |
|---|---|
| settings | `Wscratch_Gnone`, `Wscratch_Gfroze` |
| primary quantity | Δ = Acc(frozen source filter) − Acc(no filter), paired by seed |
| budget | n = 100 sentences (80 train / 20 dev) sampled from a fixed pool (`--pool_size 200`) |
| test, targets with a train split (nds, oc) | the official UD test split; the pool comes from the start of train |
| test, test-only treebanks (bho, hsb, scn, yue) | pool = the first min(200, total − 100) sentences; test = the sentences after it, at most 2000 (`--min_test 100 --max_test 2000`) |
| labels | the new probe predicts all 37 coarse UD relations (`--target_labels`), so every source is scored on exactly the same words |
| learning rate | each probe is fit at lr ∈ {3e-4, 1e-3, 3e-3, 1e-2} (`--lr_grid`); the lowest dev loss is kept, and the result records the choice and every dev loss |
| initialization | torch and numpy are reseeded before every fit, so both settings and all sources start from the same probe with the same batch order |
| seeds | 42–46 |
| stopping | as in v3 (up to 100 epochs, `--early_stop 2 --min_delta 1e-3`) |

### Submit and analyse

```bash
bash direction1/scripts/submit_pairs.sh --dry-run   # 2 models x 7 sources x 5 seeds = 70 jobs
bash direction1/scripts/submit_pairs.sh             # skips any source missing from results_index.json
python -m direction1.transfer.analyze_pairs         # pair tables, sign-flip test, source x target Δ heatmap
python -m direction1.transfer.plot_pair_accuracy    # accuracy per target: no filter vs frozen filters
python -m direction1.transfer.plot_pair_gates       # what distinguishes helpful source filters
```

Results go to `direction1/transfer/results_pairs_n100/`, and figures to `direction1/transfer/figures_pairs/`.

**Sanity check.** `analyze_pairs` prints the spread of the no-filter accuracy across sources for the same target and seed. It must be 0.000 (up to GPU nondeterminism), because that probe never sees the source.

---

## 8. Open protocol checks

These two checks decide the learning rate and schedule for the next, larger grid. Run them before any new large-scale runs.

```bash
# XLM-R initial learning rate: v3 used 1e-3, and many XLM-R probes hit the 100-epoch cap
for lr in 3e-3 1e-2; do
  MODELS=FacebookAI/xlm-roberta-base LR=$lr SETTINGS=train \
  OUTPUT_DIR=direction1/transfer/results_lr/lr$lr bash direction1/scripts/submit_transfer_v3.sh
done

# constant lr vs ReduceLROnPlateau on a random sample of cells (both from lr 1e-3; same initial probe and batch order)
bash direction1/scripts/submit_schedule_check.sh    # 24 jobs; the sample is saved to results_lr/schedule_check/selection.json
```

- **Constant lr:** `--decay_rate 1` turns the schedule off.
- **Comparing:** compare `test_accuracy` between `results_lr/schedule_check/plateau` and `.../constant`, cell by cell. Compare stop epochs from the `Early stop at epoch` lines in the job logs.

---

## 9. Tests

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m unittest discover -s tests   # 44 tests, CPU only
```

The tests cover:
- filter gating;
- label-weighted accuracy;
- the setting matrix, including the label sets and the reseeded initial probe;
- the fixed-budget split.

If `python -m unittest tests/...` cannot find `tests.*`, an installed package named `tests` is shadowing the folder. Use `discover -s tests` as above.

---

## 10. Pitfalls we hit

- **Decoder out-of-memory.** Keep the embedding batch at 8 for Aya and Qwen on 48 GB GPUs. The submit scripts already do this; probe training stays at batch 32.
- **Shared filesystems.** Concurrent seed jobs create the same result directories. `save_result` retries the write, so a rare "file not found" during saving is not fatal. If a job still dies, resubmit it; finished cells are skipped.
- **GPU architecture.** "CUDA error: no kernel image is available" means the installed torch build does not support that GPU. Install a matching CUDA build, or avoid those nodes.
- **Duplicate source runs.** The index keeps the best-accuracy run per key, which can silently pick a different probe (see step 1).
- **Gated model.** Aya needs Hugging Face access approval before the first download.
- **The `bug` analysis section** needs grids that are not part of this replication; skip it.

---

## 11. File map

| path | role |
|---|---|
| `direction1/classify.py` | trains a source probe (with or without a filter) |
| `models/encoders.py` | `PrismEncoder`: LM embeddings, word pooling, DCT filter (`frq_logits` = apply the sigmoid gate) |
| `direction1/transfer/zero_shot.py` | setting names, `build_setting`, data splits, `train_probe`, `evaluate` |
| `direction1/transfer/run_batch.py` | batch runner: one LM load per job, cached embeddings, all settings × targets |
| `direction1/scripts/slurm_transfer.sh` | SLURM wrapper around `run_batch` (positional args, then extra flags) |
| `direction1/scripts/submit_transfer_v3.sh`, `submit_pairs.sh`, `submit_schedule_check.sh` | job grids |
| `train_source.slurm` | SLURM wrapper around `classify.py` for source probes |
| `direction1/update_results_index.py` | builds `results_index.json` (source-run lookup) |
| `direction1/transfer/prepare_coverage_data.py`, `prepare_pair_data.py` | target data and target lists |
| `direction1/transfer/related_pairs.json`, `pair_targets.json`, `coverage_targets.json` | experiment configs |
| `direction1/transfer/data_checksums.md5` | checksums of every input data file |
| `direction1/transfer/analyze_v3.py`, `plot_v3.py` | v3 analysis and figures |
| `direction1/transfer/analyze_pairs.py`, `plot_pair_accuracy.py`, `plot_pair_gates.py` | related-pair analysis and figures |
| `direction1/transfer/plot_heatmaps.py`, `make_transfer_table.py`, `scripts/submit_transfer.sh` | the earlier v2 grid (paper draft); kept for reference |
