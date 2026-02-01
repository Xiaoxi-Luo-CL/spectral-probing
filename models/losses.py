import torch
import torch.nn as nn

#
# Loss Functions
#


class LabelLoss(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self._xe_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self._classes = classes
        self._class2id = {c: i for i, c in enumerate(self._classes)}

    def __repr__(self):
        return \
            f'<{self.__class__.__name__}: loss=XEnt, num_classes={len(self._classes)}>'

    def _gen_target_labels(self, logits, targets):
        target_labels = torch.tensor(
            [self._class2id[c] for c in targets], dtype=torch.long,
            device=logits.device
        )  # (num_targets,)

        return target_labels

    def forward(self, logits, targets):
        '''logit: (batch_size, seq_len, class_num)
        target: list of str labels, number of elements = non-inf logits rows'''
        # gather target labels
        target_labels = self._gen_target_labels(logits, targets)
        # flatten logits
        # (sum_over_seq_lens,)
        flat_logits = logits[torch.sum(logits, dim=-1) != float('-inf'), :]

        return self._xe_loss(flat_logits, target_labels)

    def get_accuracy(self, logits, targets):
        # gather target labels
        target_labels = self._gen_target_labels(logits, targets)
        # get labels from logits
        # (sum_over_seq_lens,)
        flat_logits = logits[torch.sum(logits, dim=-1) != float('-inf'), :]
        labels = torch.argmax(flat_logits, dim=-1)

        # compute label accuracy
        num_label_matches = torch.sum(labels == target_labels)
        accuracy = float(num_label_matches / labels.shape[0])

        return accuracy


class RegressionLoss(nn.Module):
    def __init__(self, classes=None):
        super().__init__()
        self._mse_loss = nn.MSELoss(reduction='mean')

    def __repr__(self):
        return f'<{self.__class__.__name__}: loss=MSE, acc=MSE>'

    def _label2float(self, targets, device):
        '''convert str labels to float'''
        try:
            values = [float(t) for t in targets]
        except ValueError:
            raise ValueError(
                "Targets for regression must be convertable to floats.")
        return torch.tensor(values, dtype=torch.float, device=device)

    def forward(self, logits, targets):
        mask = ~torch.isnan(logits)
        flat_logits = logits[mask].squeeze()
        target_values = self._label2float(
            targets, device=logits.device)

        return self._mse_loss(flat_logits, target_values)

    def get_accuracy(self, logits, targets):
        '''return mean square error as accuracy'''
        return self.forward(logits, targets)
