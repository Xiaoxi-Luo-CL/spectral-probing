### wikiann, NER
token-level, must repeat labels
```
python classify.py tasks/wikiann/en-train.csv tasks/wikiann/en-dev.csv --repeat_labels bert-base-cased --embedding_caching "auto(512)" --classifier linear first-exp/ --random_seed 42 --prediction
```
### UD, POS
token-level, must repeat labels
```
python classify.py tasks/ud-syntax/en-ewt-gum-train-pos.csv tasks/ud-syntax/en-ewt-gum-dev-pos.csv --repeat_labels bert-base-cased --embedding_caching "auto(512)" --classifier linear results/bert-ud-pos/ --random_seed 42
```
### UD, dependency parsing (original)
token-level, must repeat labels
```
python classify.py tasks/ud-syntax/dep/en-ewt-gum-train-relations.csv tasks/ud-syntax/dep/en-ewt-gum-dev-relations.csv --repeat_labels bert-base-cased --embedding_caching "auto(512)" --classifier linear --random_seed 42
```
### UD, dependency parsing, predict relative position
token-level, must repeat labels. 
Regression!
```
python classify.py tasks/ud-syntax/position/en-ewt-gum-train-position.csv tasks/ud-syntax/position/en-ewt-gum-dev-position.csv --repeat_labels bert-base-cased --embedding_caching "auto(512)" --classifier linear --random_seed 42 --loss_type regression --lr 5e-3
```
relative position prediction. Regression!
```
python classify.py tasks/ud-syntax/relative_position/en-ewt-gum-train-relative_position.csv tasks/ud-syntax/relative_position/en-ewt-gum-dev-relative_position.csv --repeat_labels bert-base-cased --embedding_caching "auto(512)" --classifier linear --random_seed 42 --loss_type regression --learning_rate 5e-3
```
### XNLI, Language Inference
sentence-level
```
python classify.py tasks/xnli/en-train.csv tasks/xnli/en-dev.csv gpt2 --embedding_caching "band(512, 0, 33)" --classifier linear --random_seed 42 --embedding_pooling last
```
