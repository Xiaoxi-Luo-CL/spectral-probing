import sys

import torch
import numpy as np

from collections import defaultdict


def truncate_label_for_word_pooling(inputs, flat_labels, encoder):
    """
    Truncate token-level labels to match the number of words that survive
    tokenizer truncation when encoder.pool_word_embed is enabled.

    Args:
        inputs: list[str]
            Raw batch sentences.
        flat_labels: list
            Flattened token labels for the whole batch, in sentence order.
        encoder: PrismEncoder
            Must expose tokenize(...) and pool_word_embed.

    Returns:
        list
            Flattened labels aligned to encoder outputs after word pooling.
    """
    if encoder.pool_word_embed is None:
        return flat_labels

    tok_inputs = encoder.tokenize(
        [sentence.split(' ') for sentence in inputs], tokenized=True)
    truncated_labels = []
    label_cur = 0

    for sidx, sentence in enumerate(inputs):
        orig_word_num = len(sentence.split(' '))
        sentence_labels = flat_labels[label_cur:label_cur + orig_word_num]
        label_cur += orig_word_num
        word_ids = tok_inputs.word_ids(batch_index=sidx)

        # Count distinct surviving words after truncation, ignoring specials/padding.
        surviving_word_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                continue
            if word_id != previous_word_id:
                surviving_word_ids.append(word_id)
                previous_word_id = word_id

        num_surviving_words = len(surviving_word_ids)
        if num_surviving_words < orig_word_num:
            print(
                f"Warning: Truncating {orig_word_num - num_surviving_words} words from sentence '{sentence}' for word pooling.")
        truncated_labels.extend(sentence_labels[:num_surviving_words])

    return truncated_labels


def classify_dataset(
        classifier, criterion, optimizer,
        dataset, batch_size, mode='train',
        repeat_labels=False, return_predictions=False):
    stats = defaultdict(list)

    # set model to training mode
    if mode == 'train':
        classifier.train()
        batch_generator = dataset.get_shuffled_batches
    # set model to eval mode
    elif mode == 'eval':
        classifier.eval()
        batch_generator = dataset.get_batches

    # iterate over batches
    for bidx, batch_data in enumerate(batch_generator(batch_size)):
        # set up batch data
        sentences, labels, num_remaining = batch_data
        # repeat labels
        if repeat_labels:
            labels = dataset.repeat_batch_labels(
                sentences, labels, classifier._emb)

        label_type = dataset.get_label_level()
        if label_type == 'token' and classifier._emb.pool_word_embed is not None:
            labels = truncate_label_for_word_pooling(
                sentences, labels, classifier._emb
            )

        # when training, perform both forward and backward pass
        if mode == 'train':
            # zero out previous gradients
            optimizer.zero_grad()
            # forward pass, ['labels', 'logits', 'flat_logits']
            # print('Performing forward pass...')
            predictions = classifier(sentences, label_type=label_type)
            # propagate loss
            loss = criterion(predictions['logits'], labels)
            loss.backward()
            optimizer.step()

        # when evaluating, perform forward pass without gradients
        elif mode == 'eval':
            with torch.no_grad():
                # forward pass
                predictions = classifier(sentences, label_type=label_type)
                # calculate loss
                loss = criterion(predictions['logits'], labels)

        # calculate accuracy
        accuracy = criterion.get_accuracy(
            predictions['logits'].detach(), labels)

        # store statistics
        stats['loss'].append(float(loss.detach()))
        stats['accuracy'].append(float(accuracy))

        # store predictions
        if return_predictions:
            # iterate over inputs items
            for sidx in range(predictions['labels'].shape[0]):
                # append non-padding predictions as list
                predicted_labels = predictions['labels'][sidx]
                stats['predictions'].append(
                    predicted_labels[predicted_labels != -1].tolist())

        # print batch statistics
        pct_complete = (1 - (num_remaining / len(dataset._inputs))) * 100
        sys.stdout.write(
            f"\r[{mode.capitalize()} | Batch {bidx + 1} | {pct_complete:.2f}%] "
            f"Acc: {np.mean(stats['accuracy']):.4f}, Loss: {np.mean(stats['loss']):.4f}\n"
        )
        sys.stdout.flush()

    # clear line
    print("\r", end='')

    return stats
