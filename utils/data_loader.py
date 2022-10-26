from torch.utils import data
import json
from transformers import AutoTokenizer
from utils.const import POLARITY, MODALITY, INTENSITY, TRIGGERS, PRICE
from utils.helper_functions import build_vocab

model = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
new_tokens = POLARITY + MODALITY + INTENSITY + TRIGGERS
num_added_toks = tokenizer.add_tokens(new_tokens)
#model.resize_token_embeddings(len(tokenizer))


all_price, price2idx, idx2price = build_vocab(PRICE)

class CommodityPriceDataset(data.Dataset):
    """ A Module to read in dataset in the form of .json file """
    def __init__(self, fpath, content_scope):
        
        self.tokenizer = tokenizer
        self.header, self.body, self.movement_label, self.WTI_difference, self.Brent_difference = [], [], [], [], []

        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                if content_scope == 'all':
                    header = item['header']
                    body = ' '.join(item['body'])
                elif content_scope == 'minus_properties':
                    header_trigger, header_properties, header_arguments = split_details(item['header'])
                    header = header_trigger + ' ' + header_arguments
                    
                    for b in body:
                        trigger, properties, arguments = split_details(b)
                        body += trigger + ' ' + arguments
                        
                elif content_scope == 'minus_arguments':
                    header_trigger, header_properties, header_arguments = split_details(item['header'])
                    header = header_trigger + ' ' + header_properties
                    
                    for b in body:
                        trigger, properties, arguments = split_details(b)
                        body += trigger + ' ' + properties
                elif content_scope == 'minus_arguments_properties':
                    header_trigger, header_properties, header_arguments = split_details(item['header'])
                    header = header_trigger 
                    
                    for b in body:
                        trigger, properties, arguments = split_details(b)
                        body += trigger
                
                #### Price Movement
                self.header.append(header)
                self.body.append(body)
                #price_movement = price2idx[find_dictkey(price_dict, item['effect_event']['event_type'])]                
                price_movement = price2idx[item['wti_label']]
                self.movement_label.append(price_movement)
                self.WTI_difference.append(item['brent_difference'])
                self.Brent_difference.append(item['wti_difference'])
                
    def __len__(self):
        return len(self.movement_label)

    def __getitem__(self, idx):
        header, body, movement_label, WTI_difference, Brent_difference = \
                    self.header[idx], self.body[idx], self.movement_label[idx], self.WTI_difference[idx], self.Brent_difference[idx]
        
      
        header_input = tokenizer(header, padding='max_length', max_length = 100,
                               truncation=True, return_tensors="pt")
        body_input = tokenizer(body, padding='max_length', max_length = 412, 
                               truncation=True, return_tensors="pt")
        return header_input, body_input, movement_label, WTI_difference, Brent_difference

          
def split_details(details):
    split_details = details.split(', ')
    trigger = split_details[0]
    properties = split_details[1:4]
    arguments = split_details[4:]
    
    return trigger, ' '.join(properties), ' '.join(arguments)
