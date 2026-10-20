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
        # Unknown labels (e.g. target-language-specific tags) map to -1,
        # which is the ignore_index for CrossEntropyLoss and is excluded from accuracy computation below.
        target_labels = torch.tensor(
            [self._class2id.get(c, -1) for c in targets], dtype=torch.long,
            device=logits.device
        )  # (num_targets,)
        # original: [self._class2id[c] for c in targets]

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

    def get_accuracy(self, logits, targets, return_count=False):
        # gather target labels
        target_labels = self._gen_target_labels(logits, targets)
        # get labels from logits
        # (sum_over_seq_lens,)
        flat_logits = logits[torch.sum(logits, dim=-1) != float('-inf'), :]
        labels = torch.argmax(flat_logits, dim=-1)

        assert target_labels.shape == labels.shape, f"target_labels shape {target_labels.shape} does not match labels shape {labels.shape}"

        # exclude tokens with unknown labels (mapped to -1) from accuracy
        # ------ original ------:
        # num_label_matches = torch.sum(labels == target_labels)
        # accuracy = float(num_label_matches / labels.shape[0])
        known = target_labels != -1
        num_known = int(known.sum())
        if num_known == 0:
            return (0.0, 0) if return_count else 0.0
        num_label_matches = torch.sum(labels[known] == target_labels[known])
        accuracy = float(num_label_matches / num_known)

        # the count lets callers weight each batch by how many labels it scored
        return (accuracy, num_known) if return_count else accuracy


class RegressionLoss(nn.Module):
    def __init__(self, classes=None, round_acc=False):
        super().__init__()
        self._mse_loss = nn.MSELoss(reduction='mean')
        self.round_acc = round_acc

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
        target_values = self._label2float(targets, logits.device)
        return self._mse_loss(flat_logits, target_values)

    def get_accuracy(self, logits, targets, return_count=False):
        '''return accuracy after rounding to nearest integer'''
        num_scored = int((~torch.isnan(logits)).sum())
        if self.round_acc:
            mask = ~torch.isnan(logits)
            flat_logits = logits[mask].squeeze()
            target_tensor = self._label2float(targets, logits.device)
            # as long as the prediction is within [3.5, 4.5), we think it is 4
            preds = torch.round(flat_logits)
            correct = (preds == target_tensor).float()
            accuracy = correct.mean().item()
        else:
            accuracy = self.forward(logits, targets)
        return (accuracy, num_scored) if return_count else accuracy
