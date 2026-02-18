## Supported Models & Languages
+ Languages (${lan}): zh, ja, es, fr, de, en
+ Models (${model}): `bert-base-multilingual-cased`, `FacebookAI/xlm-roberta-base (XLM-R)`, `bigscience/bloom-560m`, `bigscience/bloom-1b1`, `bigscience/bloom-1b7`, `bigscience/bloom-3b`, `bigscience/bloom-7b1`, `mistralai/Mistral-7B-v0.1`

## Tasks
Run the `classify.py` command with parameters (see below) to train and evaluate probes. Replace ${lan} and ${model} with your desired language code and model identifier.

The following **classification tasks** use standard Cross-Entropy Loss. The probe predicts discrete labels for each token.

1. Named Entity Recognition (WikiANN)

```bash
python classify.py \
    tasks/wikiann/${lan}-train.csv \
    tasks/wikiann/${lan}-dev.csv \
    --repeat_labels \
    --model ${model} \
    --embedding_caching "auto(512)" \
    --classifier linear \
    --random_seed 42
```

2. UD POS Tagging. Note that we use `gsd` treebank for all languages except English. For English, we use `ewt` treebank (change the "gsd" below to "ewt").
```bash
python classify.py \
    tasks/ud-syntax/${lan}/pos/${lan}-gsd-train-pos.csv \
    tasks/ud-syntax/${lan}/pos/${lan}-gsd-dev-pos.csv \
    --repeat_labels \
    --model ${model} \
    --embedding_caching "auto(512)" \
    --classifier linear \
    --random_seed 42
```

3. Dependency Relation Classification
```bash
python classify.py \
    tasks/ud-syntax/${lan}/relations/${lan}-ewt-train-relations.csv \
    tasks/ud-syntax/${lan}/relations/${lan}-ewt-dev-relations.csv \
    --repeat_labels \
    --model ${model} \
    --embedding_caching "auto(512)" \
    --classifier linear \
    --random_seed 42
```

The following **regression tasks** use MSE loss, and predict continuous values (e.g., token indices or relative distances for each token. `--loss_type regression`` is required.

4. Absolute Head Position Prediction
Predict the exact index of the syntactic head for each token.

```bash
python classify.py \
    tasks/ud-syntax/${lan}/position/${lan}-ewt-train-position.csv \
    tasks/ud-syntax/${lan}/position/${lan}-ewt-dev-position.csv \
    ${model} \
    --embedding_caching "auto(512)" \
    --classifier linear \
    --random_seed 42 \
    --repeat_labels \
    --loss_type regression
```

5. Relative Head Position Prediction
Predict the distance between the current token and its head.

```bash
python classify.py \
    tasks/ud-syntax/${lan}/relative_position/${lan}-ewt-train-relative_position.csv \
    tasks/ud-syntax/${lan}/relative_position/${lan}-ewt-dev-relative_position.csv \
    ${model} \
    --embedding_caching "auto(512)" \
    --classifier linear \
    --random_seed 42 \
    --repeat_labels \
    --loss_type regression
```