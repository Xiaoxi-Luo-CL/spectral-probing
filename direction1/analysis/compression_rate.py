import transformers
import pandas as pd
from tqdm import tqdm


def calculate_compression_rates(data_dict, models):
    """
    Calculates subword fertility (N_subwords / N_gold_words) across datasets and models.
    data_dict: dict mapping language codes to file paths (e.g., {'zh': 'zh_data.csv'})
    """
    results = {}

    for lm_name in models:
        results[lm_name] = {}
        tokenizer_kwargs = {'use_fast': True}

        if any(m in lm_name.lower() for m in ['gpt2', 'bloom', 'mistral', 'aya']):
            tokenizer_kwargs['add_prefix_space'] = True

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            lm_name, **tokenizer_kwargs)

        for lang, path in data_dict.items():
            df = pd.read_csv(path)

            total_gold_words = 0
            total_subwords = 0

            for text in df['text']:
                if not isinstance(text, str):
                    continue

                total_gold_words += len(text.split())
                # Exclude special tokens (CLS, SEP, BOS, EOS) for strict ratio calculation
                subwords = tokenizer.tokenize(text, add_special_tokens=False)
                total_subwords += len(subwords)

            rate = total_subwords / total_gold_words if total_gold_words > 0 else 0
            results[lm_name][lang] = round(rate, 3)

    return pd.DataFrame(results).T


def calculate_premium(data_dict, models):
    """
    Calculates subword fertility (N_subwords / N_gold_words) across datasets and models.
    data_dict: dict mapping language codes to file paths (e.g., {'zh': 'zh_data.csv'})
    """
    results = {}

    for lm_name in models:
        results[lm_name] = {}
        tokenizer_kwargs = {'use_fast': True}
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            lm_name, **tokenizer_kwargs)

        for lang, path in data_dict.items():
            df = pd.read_csv(path)
            total_subwords = 0
            for text in df['text']:
                if 'bloom' in lm_name and lang == 'zh':
                    from IPython import embed
                    embed()
                if not isinstance(text, str):
                    continue
                if lang == 'zh':
                    sentence = ''.join(text.split(' '))
                else:
                    sentence = text
                subwords = tokenizer.tokenize(
                    sentence, add_special_tokens=False)
                total_subwords += len(subwords)
            results[lm_name][lang] = total_subwords

        # compare against English baseline
        for lang in results[lm_name]:
            if lang != 'en':
                premium = results[lm_name][lang] / results[lm_name]['en']
                results[lm_name][lang] = round(premium, 3)

        for lang in results[lm_name]:
            results[lm_name]['en'] = 1.0
    return pd.DataFrame(results).T


# Example execution
if __name__ == "__main__":
    models = [
        "bert-base-multilingual-cased",
        "FacebookAI/xlm-roberta-base",
        "bigscience/bloom-560m",
        "bigscience/bloom-7b1",
        "CohereForAI/aya-expanse-8b",
        "mistralai/Mistral-7B-v0.1"
    ]

    data_paths = {
        'de': 'tasks/ud-syntax/de/pos/de-gsd-train-pos.csv',
        'en': 'tasks/ud-syntax/en/pos/en-ewt-train-pos.csv',
        'es': 'tasks/ud-syntax/es/pos/es-gsd-train-pos.csv',
        'fr': 'tasks/ud-syntax/fr/pos/fr-gsd-train-pos.csv',
        'ja': 'tasks/ud-syntax/ja/pos/ja-gsd-train-pos.csv',
        'zh': 'tasks/ud-syntax/zh/pos/zh-gsd-train-pos.csv',
        'fi': 'tasks/ud-syntax/fi/pos/fi-tdt-train-pos.csv',
        'hi': 'tasks/ud-syntax/hi/pos/hi-hdtb-train-pos.csv',
        'tr': 'tasks/ud-syntax/tr/pos/tr-kenet-train-pos.csv',
        'vi': 'tasks/ud-syntax/vi/pos/vi-vtb-train-pos.csv',
        'ko': 'tasks/ud-syntax/ko/pos/ko-kaist-train-pos.csv',
        'ur': 'tasks/ud-syntax/ur/pos/ur-udtb-train-pos.csv',
    }

    df_results = calculate_compression_rates(data_paths, models)
    df_results.to_csv('analysis/compress_rate.csv')
    # df_premium = calculate_premium(data_paths, models)
    # df_premium.to_csv('analysis/premium_rate.csv')
