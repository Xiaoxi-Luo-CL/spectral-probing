for lan in "fr" "zh"
do
    for model in "bert-base-multilingual-cased" "FacebookAI/xlm-roberta-base" "bigscience/bloom-560m"
    do
        sbatch slurm.sh ${lan} ${model}
    done
done