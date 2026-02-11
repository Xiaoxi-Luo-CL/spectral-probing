import logging
import os
import re
import sys
import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np


def create_save_dir(folder='', suffix=''):
    """
    Create a new directory with the current date and time as its name and return the path of the new directory.
    """
    subFolderName = re.sub(
        r'[^0-9]', '', str(datetime.datetime.now()))[4:16]+suffix
    path = os.path.join(folder, subFolderName)
    os.makedirs(path, exist_ok=True)
    # my_mkdir(os.path.join(path, 'output'))
    return path


def setup_experiment(out_path, prediction=False, log_name='classify.log', mode='a'):
    # setup logging
    log_format = '%(message)s'
    log_level = logging.INFO
    logging.basicConfig(filename=os.path.join(
        out_path, log_name), filemode=mode,
        format=log_format, level=log_level, force=True)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    if not os.path.exists(out_path):
        if prediction:
            logging.info(
                f"Experiment path '{out_path}' does not exist. Cannot run prediction. Exiting.")
            exit(1)

        # if output dir does not exist, create it (new experiment)
        logging.info(f"Path '{out_path}' does not exist. Creating...")
        os.mkdir(out_path)
    # if output dir exist, check if predicting
    # else:
    #     # if not predicting, verify overwrite
    #     if not prediction:
    #         response = None

    #         while response not in ['y', 'n']:
    #             response = input(
    #                 f"Path '{out_path}' already exists. Overwrite? [y/n] ")
    #         if response == 'n':
    #             exit(1)


def setup_filter(filter_id, frq_init='ones', frq_noise=0.1, frq_scale=1.0):
    from models.encoders import PrismEncoder

    # set up filter definitions
    filter_match = re.match(
        r'(?P<name>[a-z]+?)(?P<args>\((\d+,? *)*\))', filter_id, flags=re.IGNORECASE)
    args_pattern = re.compile(r'(\d+),?')

    # check if filter syntax is correct
    if filter_match:
        filter_args = [int(a)
                       for a in args_pattern.findall(filter_match['args'])]
        # parse nofilter
        if filter_match['name'] == 'nofilter':
            if len(filter_args) != 0:
                logging.error(
                    f"[Error] nofilter does not take any arguments. Exiting.")
                exit(1)
            return None
        # parse eqalloc filter
        elif filter_match['name'] == 'eqalloc':
            if len(filter_args) != 3:
                logging.error(
                    f"[Error] eqalloc filter requires three arguments 'max_freqs', 'num_bands', 'band_idx'. Exiting.")
                exit(1)
            return PrismEncoder.gen_equal_allocation_filters(filter_args[0], filter_args[1])[filter_args[2]]
        # parse band filter
        elif filter_match['name'] == 'band':
            if len(filter_args) != 3:
                logging.error(
                    f"[Error] band filter requires three arguments 'max_freqs', 'start_idx', 'end_idx'. Exiting.")
                exit(1)
            return PrismEncoder.gen_band_filter(filter_args[0], filter_args[1], filter_args[2])
        elif filter_match['name'] == 'auto':
            if len(filter_args) != 1:
                logging.error(
                    f"[Error] auto requires the argument 'max_freqs'. Exiting.")
                exit(1)
            return PrismEncoder.gen_auto_filter_initialization(filter_args[0], frq_init, frq_noise, frq_scale)
        else:
            logging.error(
                f"[Error] Unknown filter type '{filter_match['name']}'. Exiting.")
            exit(1)
    else:
        logging.error(
            f"[Error] Could not parse filter '{filter_id}'. Exiting.")
        exit(1)


def analyze_filter(filter, folder, fig_name):
    weight = torch.sigmoid(filter)

    # draw the weight plot
    plt.ylim(0, 1)
    plt.plot(weight.numpy())
    plt.title('Frequency Filter Weights')
    plt.xlabel('Frequency')
    plt.ylabel('Weight')
    plt.grid()
    plt.savefig(folder + '/' + fig_name)
    plt.clf()
    # treat weights as a distribution and analyze entropy
    distribution = torch.softmax(filter, dim=0).numpy()
    # save distribution
    np.save(folder + '/distribution.npy', distribution)
    # compute entropy
    entropy = -np.sum(distribution * np.log(distribution + 1e-12))

    logging.info(
        f"\nFilter analysis results saved to {folder}/{fig_name} and distribution.npy")
    logging.info(f"Entropy: {entropy}")


def check_args(args):
    '''Check if arguments have no conflict. If not, return a directory name based on the arguments.'''
    if 'relative' in args.train_path or 'position' in args.train_path:
        assert args.loss_type == 'regression', \
            "[Error] Relative/position embeddings can only be used with label classification tasks. Exiting."

    if not args.repeat_labels:
        assert args.embedding_pooling is not None, \
            "Embedding pooling must be specified when repeat_labels is False."
    if args.repeat_labels and args.embedding_pooling is not None:
        logging.warning(
            "repeat_labels=True with embedding_pooling is not common practice.")
    if args.prediction:
        assert args.out_path is not None, "Output path must be specified when running prediction."
        # assert path exists
        assert os.path.exists(
            args.out_path), f"Output path '{args.out_path}' does not exist."


def dir_name(args):
    if '/' in args.lm_name:
        model = args.lm_name.split('/')[-1]
    else:
        model = args.lm_name
    task_name = args.train_path.split('/')[-2]
    lan = args.train_path.split('/')[-1].split('-')[0]
    return '_' + model + '_' + task_name + '_' + lan, task_name, lan
