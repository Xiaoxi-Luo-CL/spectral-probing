#!/bin/bash
# Constant lr vs ReduceLROnPlateau on a random sample of v3 cells. Per model, a fixed-seed draw picks one source,
# 4 targets and 3 trained settings (written to $OUT/selection.json); each cell runs with both schedules from the
# same initial lr, seeds 42-44. Both use the reseeded path (--lr_grid with one rate), so the two schedules start from
# the same probe and see the same batch order: only the schedule differs. Otherwise the v3 protocol.
#
# Environment overrides (defaults in brackets):
#   LR [1e-3]   SEEDS ["42 43 44"]   PICK_SEED [0]   OUT [direction1/transfer/results_lr/schedule_check]
#
# Usage (from spectral-probing/):
#   bash direction1/scripts/submit_schedule_check.sh [--dry-run]

set -euo pipefail

SLURM_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm_transfer.sh"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true
LR="${LR:-1e-3}"
OUT="${OUT:-direction1/transfer/results_lr/schedule_check}"
read -r -a SEED_LIST <<< "${SEEDS:-42 43 44}"
mkdir -p "$OUT"

# draw the sample: one line per model, "model src pool embed_batch settings target_list"
mapfile -t PICKS < <(/u501/x25luo/.conda/envs/grounding/bin/python - "$OUT" "${PICK_SEED:-0}" <<'EOF'
import json, sys
import numpy as np
from direction1.transfer.zero_shot import TRAIN_SETTINGS
out, rng = sys.argv[1], np.random.default_rng(int(sys.argv[2]))
targets = json.load(open('direction1/transfer/coverage_targets.json'))
models = {'FacebookAI/xlm-roberta-base': ('avg', 32), 'bert-base-multilingual-cased': ('avg', 32),
          'CohereLabs/aya-expanse-8b': ('last', 8), 'Qwen/Qwen2.5-7B': ('last', 8)}
sel = {}
for m, (pool, eb) in models.items():
    src = str(rng.choice(['en', 'zh']))
    tg = [targets[i] for i in sorted(rng.choice(len(targets), 4, replace=False))]
    st = [str(x) for x in rng.choice(TRAIN_SETTINGS, 3, replace=False)]
    path = f'{out}/targets_{m.split("/")[-1]}.json'
    json.dump(tg, open(path, 'w'), indent=2)
    sel[m] = {'src': src, 'targets': [t['lang'] for t in tg], 'settings': st}
    print(m, src, pool, eb, ','.join(st), path)
json.dump(sel, open(f'{out}/selection.json', 'w'), indent=2)
EOF
)

total=0
for line in "${PICKS[@]}"; do
    read -r model src pool embed_batch settings target_list <<< "$line"
    echo "$(basename "$model"): src=$src settings=$settings targets=$target_list"
    for sched in plateau constant; do
        decay=0.5; [[ "$sched" == constant ]] && decay=1
        for seed in "${SEED_LIST[@]}"; do
            name="sched_$(basename "$model")_${sched}_s${seed}"
            extra=(--epochs 100 --early_stop 2 --min_delta 1e-3 --largest_n_only --lr_grid "$LR" --decay_rate "$decay")
            if $DRY_RUN; then
                echo "[DRY-RUN] $name"
            else
                echo "Submitting: $name"
                sbatch --job-name="$name" "$SLURM_SCRIPT" "$model" "$src" "$pool" relations "$settings" \
                    100,500,1000 "$target_list" "$OUT/$sched" "$seed" "$embed_batch" "${extra[@]}"
            fi
            total=$((total + 1))
        done
    done
done
echo "jobs: $total"
