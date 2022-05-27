
import random

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_metrics(outputs, labels, type='binary'):
    if type == 'binary':
        proba = torch.sigmoid(outputs)
        pred = (proba > 0.5)

        correct = pred.eq(outputs.bool()).sum().item()
        return {
            'auc': roc_auc_score(labels, proba),
            'correct': correct
        }
    elif type == 'regression':
        return {
            'mse': torch.nn.functional.mse_loss(outputs, labels, reduction='mean'),
            'l1': torch.nn.functional.l1_loss(outputs, labels, reduction='mean')
        }