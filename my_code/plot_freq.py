import torch
import torch.nn as nn
from models.encoders import PrismEncoder, load_pooling_function
from models.classifiers import load_classifier
from utils.datasets import LabelledDataset
from utils.setup import setup_filter
import argparse
import os
import matplotlib.pyplot as plt


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Classifier Training')
    arg_parser.add_argument('--model_folder')
    arg_parser.add_argument('--valid_path', help='path to validation data')
    arg_parser.add_argument(
        '--filter', help='type of frequency filter to apply')
    arg_parser.add_argument(
        '--embedding_tuning', action='store_true', default=False, help='set flag to tune the full model including embeddings (default: False)')
    arg_parser.add_argument(
        '--embedding_pooling', choices=['mean', 'last', None],
        default=None, help='embedding pooling strategy (default: None)')
    arg_parser.add_argument(
        '--fig_name', default='spectral_profile.png', type=str, help='name of the figure')
    arg_parser.add_argument('--classifier', default='linear')
    return arg_parser.parse_args()


def main():
    args = parse_arguments()
    frq_filter = setup_filter(args.filter)
    frq_tuning = (args.filter != 'nofilter()')
    # prediction

    model_path = os.path.join(args.model_folder, 'best.pt')
    if not os.path.exists(model_path):
        print(
            f"[Error] No pre-trained model available at '{model_path}'. Exiting.")
        exit(1)

    pooling_strategy = None if args.embedding_pooling is None else load_pooling_function(
        args.embedding_pooling)

    classifier_constructor, loss_constructor = load_classifier(args.classifier)

    valid_data = LabelledDataset.from_path(args.valid_path)
    label_types = sorted(set(valid_data.get_label_types()))

    with torch.serialization.safe_globals([nn.modules.linear.Linear]):
        encoder = PrismEncoder.load(
            model_path, frq_filter=frq_filter, frq_tuning=frq_tuning,
            emb_tuning=args.embedding_tuning, emb_pooling=pooling_strategy, cache=None
        )
        classifier = classifier_constructor(
            emb_model=encoder, classes=label_types
        )
        classifier = classifier.load(model_path, emb_model=encoder)
    filter = classifier._emb._frq_filter.detach().cpu()
    weight = torch.sigmoid(filter).numpy()

    # draw the weight plot
    plt.plot(weight)
    plt.title('Frequency Filter Weights')
    plt.xlabel('Frequency')
    plt.ylabel('Weight')
    plt.grid()
    plt.savefig(args.model_folder + '/' + args.fig_name)


if __name__ == '__main__':
    main()
# python -m my_code.plot_freq --model_folder results/01142307gpt2_mkqa/ --valid_path tasks/mkqa/en-dev.csv --filter "auto(512)" --embedding_pooling last
