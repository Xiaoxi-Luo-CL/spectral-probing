'''Extract hidden representations from a trained model for a given dataset, and save them.
The saved representations can be used: (1) visualization (2) DCT.
'''
import argparse
import numpy as np
import torch
from utils.datasets import LabelledDataset
from analysis.raw_encoder import PrismEncoder
from tqdm import tqdm


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Classifier Training')
    arg_parser.add_argument('--train_path', help='path to training data',
                            default='tasks/ud-syntax/en/pos/en-ewt-train-pos.csv')
    arg_parser.add_argument(
        '--lm_name', default='bert-base-cased', help='embedding model identifier')
    arg_parser.add_argument('--batch_size', default=128, type=int)
    arg_parser.add_argument(
        '-rs', '--random_seed', type=int, default=42,
        help='seed for probabilistic components (default: None)')
    arg_parser.add_argument('--save_file', default=None)

    return arg_parser.parse_args()


def main():
    args = parse_arguments()

    # set random seeds
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.random.manual_seed(args.random_seed)

    train_data = LabelledDataset.from_path(args.train_path)
    encoder = PrismEncoder(lm_name=args.lm_name, frq_filter=None)
    batch_generator = train_data.get_batches

    embed_ls, label_ls = [], []
    for batch_data in tqdm(batch_generator(args.batch_size)):
        sentences, label, _ = batch_data
        label_type = train_data.get_label_level()
        embeddings, attentions = encoder(sentences, label_type)
        label_repeated = train_data.repeat_batch_labels(
            sentences, label, encoder)

        # Extract valid tokens sentence by sentence
        batch_size = embeddings.size(0)
        for i in range(batch_size):
            sent_embs = embeddings[i, attentions[i], :]
            embed_ls.append(sent_embs.cpu().numpy())
            # Ensure label structure matches embedding structure
            label_ls.append(label_repeated[i])

    if args.save_file is not None:
        save_path = args.save_file
    else:
        dataset = args.train_path.split('/')[-1].split('.')[0]
        save_path = f'analysis/{dataset}_{args.lm_name}_embedding.pt'

    torch.save({
        'embeddings': embed_ls,
        'labels': label_ls,
    }, save_path)


if __name__ == '__main__':
    main()
