import numpy as np
from os.path import join
import os
import torch.nn as nn

NONE = 'O'

def build_vocab(labels):
    """ build vocabulary for labels """
    all_labels = []
        
    for label in labels:
        all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}
    
    return all_labels, label2idx, idx2label