import numpy as np
import torch
from torch.nn import init
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import copy
import os
import logging
from functools import reduce
from operator import concat

from torchmetrics import MeanAbsolutePercentageError


class CommodityPrice(nn.Module):
    """ Classify Price Movement """
    def __init__(self, args, price_size, model):
                    
        super().__init__()
        
        #config_class, model_class, tokenizer_class, model_name_or_path = MODEL_CLASSES[args.transformer_type]
        hidden_dropout_prob = args.dropout
        
        #self.transformer = model_class.from_pretrained(model_name_or_path)
        self.transformer = model
        self.model_embedding_dim = 768    ### embedding size for BERT-BASE-CASED
        
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Sequential(nn.Linear(self.model_embedding_dim, price_size))
        self.regression = nn.Linear(self.model_embedding_dim, 1)   ## predict price
        
        '''
        ### Bi-LSTM Encoder
        self.rnn = nn.LSTM(bidirectional = args.bidirectional,  ### bidirectional
                           num_layers = args.n_layers,
                           input_size = self.model_input_size,
                           hidden_size = self.model_input_size//2,    
                           bias = True,
                           dropout = args.dropout if args.n_layers > 1 else 0,
                           batch_first = True)
        
        self.fc_price = nn.Sequential(nn.Linear(metadata_input_size*2, 3))
        '''
    def classify_movement(self, tokens):    
        """ Function to encode tokens with the selected Pre-trained Language Model """        
        outputs = self.transformer(input_ids = tokens)
        output = outputs[1]      ### the last hidden state
                
        #summed_output = output.mean(dim = 0)
        #print('average output: ' + str(average_output.size()))
        
        #average_output = torch.unsqueeze(average_output, 0)
        #print('output unsqueeze size: ' + str(average_output.size()))
        
        average_output = output

        average_output = self.dropout(average_output)
        price_logits = self.classifier(average_output)
        
        price_hat = price_logits.argmax(-1)    
        
        return price_logits, price_hat
    
    def regression_price(self, tokens):
        """ Function to encode tokens with the selected Pre-trained Language Model """        
        outputs = self.transformer(input_ids = tokens)
        output = outputs[1]      ### the last hidden state
        
        average_output = output
        
        average_output = self.dropout(average_output)
        price_hat = self.regression(average_output)
        
        return price_hat
    
