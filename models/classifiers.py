import torch
import torch.nn as nn

from models.encoders import *
from models.losses import *

#
# Base Classifier
#


class EmbeddingClassifier(nn.Module):
    def __init__(self, emb_model, lbl_model, classes: None | list = None):
        super().__init__()
        # internal models
        self._emb = emb_model
        self._lbl = lbl_model
        # internal variables
        self._classes = classes
        self.is_regression = (classes is None)
        # move model to GPU if available
        if torch.cuda.is_available():
            self.to(torch.device('cuda'))

    def __repr__(self):
        mode = "Regression" if self._classes is None else f"Classification ({len(self._classes)} classes)"
        return f'''<{self.__class__.__name__}:
	emb_model = {self._emb}, mode = {mode}>'''

    def get_trainable_parameters(self):
        return list(self._emb.get_trainable_parameters()) + list(self._lbl.parameters())

    def get_savable_objects(self):
        objects = self._emb.get_savable_objects()
        objects['classifier'] = self._lbl
        objects['classes'] = self._classes
        return objects

    def save(self, path):
        torch.save(self.get_savable_objects(), path)

    @staticmethod
    def load(path, emb_model, emb_pooling=None):
        objects = torch.load(path, weights_only=False)
        classes = objects['classes']
        lbl_model = objects['classifier']
        # instantiate class using pre-trained label model and fixed encoder
        return EmbeddingClassifier(
            emb_model=emb_model, lbl_model=lbl_model, classes=classes
        )

    def forward(self, sentences, label_type):
        # embed sentences (batch_size, seq_length) -> (batch_size, max_length, emb_dim)
        # if pooling, then emb_sentences: (batch_size, 1, emb_dim)
        emb_sentences, att_sentences = self._emb(sentences, label_type)

        # logits for all tokens in all sentences + padding (batch_size, max_len, num_labels)
        out_dim = 1 if self.is_regression else len(self._classes)
        init_val = float('-inf') if not self.is_regression else float('nan')

        logits = torch.ones(
            (att_sentences.shape[0],
             att_sentences.shape[1], out_dim),
            device=emb_sentences.device
        ) * init_val
        # get token embeddings of all sentences (total_tokens, emb_dim)
        emb_tokens = emb_sentences[att_sentences, :]
        # pass through classifier
        flat_logits = self._lbl(emb_tokens)  # (num_words, num_labels)
        # (batch_size, max_len, num_labels)
        logits[att_sentences, :] = flat_logits
        if self.is_regression:
            predictions = logits.squeeze(-1)
        else:
            predictions = self.get_labels(logits.detach())
        results = {
            'labels': predictions,
            'logits': logits,
            'flat_logits': flat_logits
        }

        return results

    def get_labels(self, logits):
        # get predicted labels with maximum probability (padding should have -inf)
        labels = torch.argmax(logits, dim=-1)  # (batch_size, max_len)
        # add -1 padding label for -inf logits
        labels[(logits[:, :, 0] == float('-inf'))] = -1

        return labels


#
# Classifiers
#


class LinearClassifier(EmbeddingClassifier):
    def __init__(self, emb_model, classes, bias=True):
        # instantiate linear classifier without bias
        out_dim = len(classes) if classes is not None else 1
        lbl_model = nn.Linear(emb_model.emb_dim, out_dim, bias=bias)
        super().__init__(
            emb_model=emb_model, lbl_model=lbl_model, classes=classes
        )


class MLPClassifier(EmbeddingClassifier):
    def __init__(self, emb_model, classes, hidden_dim=128, bias=True):
        out_dim = len(classes) if classes is not None else 1
        # instantiate MLP classifier
        lbl_model = nn.Sequential(
            nn.Linear(emb_model.emb_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=bias)
        )

        super().__init__(
            emb_model=emb_model, lbl_model=lbl_model, classes=classes
        )


#
# Helper Functions
#

def load_classifier(identifier, loss_type):
    if identifier == 'linear':
        classifier = LinearClassifier
    elif identifier == 'mlp':
        classifier = MLPClassifier
    else:
        raise ValueError(
            f"[Error] Unknown classifier specification '{identifier}'.")
    if loss_type == 'classification':
        loss = LabelLoss
    elif loss_type == 'regression':
        loss = RegressionLoss
    else:
        raise ValueError(
            f"[Error] Unknown loss specification '{loss_type}'.")

    return classifier, loss
