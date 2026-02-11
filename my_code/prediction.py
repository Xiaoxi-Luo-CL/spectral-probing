import argparse
import logging
import os
import json
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
# local imports
from utils.setup import (setup_experiment)
from utils.datasets import LabelledDataset
from utils.training import classify_dataset
from models.encoders import PrismEncoder
from models.classifiers import load_classifier


def shuffle_weight(orig_filter, mode='zero', para: float | None = None):
    '''Shuffle the weights of _frq_filter of the classifier
    '''
    if mode == 'reverse':
        return orig_filter.flip(0).detach().clone()
    elif mode == 'orig':
        return orig_filter.detach().clone()
    elif mode == 'shuffle':
        shuffled = orig_filter.detach().clone()
        perm = torch.randperm(shuffled.shape[0])
        return shuffled[perm]
    else:
        y = torch.sigmoid(orig_filter)
        assert type(
            para) is float, f"Parameter 'para' must be a float for mode '{mode}'."
        if mode == 'plus':
            y_target = torch.clamp(y + para, min=1e-6, max=1-1e-6)
        elif mode == 'multiply':
            y_target = torch.clamp(y * para, min=1e-6, max=1-1e-6)
        else:
            raise ValueError(f"Unknown mode '{mode}' for shuffle_weight.")

        return torch.logit(y_target)


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Classifier Training')
    # data setup
    arg_parser.add_argument('--exp_path', help='path to training data')
    arg_parser.add_argument('--perturbe_filter_mode', type=str, default=None, choices=['shuffle', 'softmax', 'plus', 'multiply'],
                            help='methods to perturbe the frequency filter, including zero, shuffle, softmax, plus, multiply')
    return arg_parser.parse_args()


def main():
    args = parse_arguments()
    exp_path = args.exp_path
    setup_experiment(exp_path, prediction=True,
                     log_name='prediction.log', mode='w')

    with open(os.path.join(exp_path, 'config.json'), 'r') as f:
        config = json.load(f)

    logging.info("===Running in prediction mode (no training).===")
    model_path = os.path.join(exp_path, 'best.pt')
    if not os.path.exists(model_path):
        logging.error(
            f"[Error] No pre-trained model available at '{model_path}'. Exiting.")
        exit(1)

    # set random seeds
    if config['random_seed'] is not None:
        np.random.seed(config['random_seed'])
        torch.random.manual_seed(config['random_seed'])

    valid_data = LabelledDataset.from_path(config['valid_path'])
    if config['loss_type'] == 'classification':
        label_types = sorted(set(valid_data.get_label_types()))
    else:
        label_types = None
    logging.info(f"Loaded {valid_data} (dev), size {len(valid_data)}.")

    from IPython import embed
    embed()  # for debugging
    encoder = PrismEncoder.load(
        model_path, frq_filter=None, frq_tuning=False,
        emb_tuning=config['embedding_tuning'], emb_pooling=None, cache=None
    )
    logging.info(f"Loaded pre-trained encoder from '{model_path}'.")

    # load classifier and loss constructors based on identifier
    classifier_constructor, loss_constructor = load_classifier(
        config['classifier'], config['loss_type'])

    # if it is regression loss but not normalized_relative
    task_name = config['train_path'].split('/')[-2]
    if config['loss_type'] == 'regression' and task_name != 'normalized_relative':
        criterion = loss_constructor(
            label_types, round_acc=True)  # type:ignore
    else:
        criterion = loss_constructor(label_types)
    logging.info(f"Using criterion {criterion}.")

    # setup classifier
    classifier = classifier_constructor(
        emb_model=encoder, classes=label_types
    )
    logging.info(f"Constructed classifier:\n{classifier}")
    # load pre-trained classifier
    with torch.serialization.safe_globals([nn.modules.linear.Linear]):
        classifier = classifier.load(
            model_path, emb_model=encoder
        )
    logging.info(f"Loaded pre-trained classifier from '{model_path}'.")

    classifier.eval()
    orig_filter = classifier._emb._frq_filter.detach().clone()

    results = defaultdict(dict)
    if args.perturbe_filter_mode is not None:
        modes = ['orig'] + [args.perturbe_filter_mode]
    else:
        modes = [('reverse', None), ('shuffle', None), ('plus', 0.1), ('plus', -0.1),
                 ('plus', -0.2), ('plus', -0.3), ('multiply', 2.0), ('multiply', 0.5), ('orig', None)]

    for (mode, para) in modes:
        perturbed = shuffle_weight(orig_filter, mode=mode, para=para)
        with torch.no_grad():
            classifier._emb._frq_filter.copy_(perturbed)

        # only predicting on validation data w/o training
        stats = classify_dataset(
            classifier, criterion, None, valid_data,
            config['batch_size'], repeat_labels=config['repeat_labels'], mode='eval',
            return_predictions=False
        )

        logging.info(
            f"Mode {mode} Prediction completed with Acc: {np.mean(stats['accuracy']):.4f}, Loss: {np.mean(stats['loss']):.4f} (mean over batches).")

        results[(mode, para)] = {
            'accuracy': np.mean(stats['accuracy']),
            'loss': np.mean(stats['loss'])
        }

    result = {str(key): value for key, value in results.items()}
    with open(os.path.join(exp_path, 'perturbed_results.json'), 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    main()
