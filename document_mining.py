import os
import json
import re
from os import listdir
from os.path import isfile, join
import spacy
import pandas as pd
import shutil 
import csv

price_dict = {'UP': ['MOVEMENT-UP-GAIN', 'GROW-STRONG', 'CAUSE-MOVEMENT-UP-GAIN'],
              'DOWN': ['MOVEMENT-DOWN-LOSS', 'SLOW-WEAK', 'CAUSE-MOVEMENT-DOWN-LOSS'],
              'FLAT': ['MOVEMENT-FLAT']
             }
             
def process_file(data, file_name):
    
    data_list = []
    
    ### process HEADER ###     
    for event in data[0]['golden-event-mentions']:  
        data_list.append(sub_process_file(event, 'H'))
        
    ### process New BODY ###
    
    for sent in data[1:]:
        if len(sent['golden-event-mentions']) > 0:
            for event in sent['golden-event-mentions']:
                data_list.append(sub_process_file(event, 'S'))
    
    #for data in data_list:
    #    if data['price'] == 'n':
    #        print(data)
    summary = restructure_data(data_list, file_name)
    return summary
    

def sub_process_file(event, sent_type):
    data_dict = {}
    data_dict['sent_type'] = sent_type
    
    data_dict['event_type'] = event['event_type']
    data_dict['trigger'] = event['trigger']
    data_dict['polarity'] = event['polarity']
    data_dict['modality'] = event['modality']
    data_dict['intensity'] = event['intensity']

   
    #### returns the movement label (UP, DOWN, STABLE) if price related.
    price, data_dict['commodity_type'], data_dict['label'], price_details = is_price(event)
    
    if price:
        data_dict['price'] = 'y'
        data_dict['arguments'] = price_details
    
    else:
        data_dict['price'] = 'n'
        data_dict['arguments'] = flatten_arguments(event['arguments'])
    
    return data_dict

def is_price(event):
    Brent, WTI, generic_price =  False, False, False
      
    movement_label = [key for key, value in price_dict.items() if event['event_type'].upper() in value]
    if len(movement_label)> 0:
        for arg in event['arguments']:
            if 'Brent' in arg['text']:
                Brent = True
                break
            elif ('WTI' in arg['text']) or ('west texas' in arg['text'].lower()) or ('new york' in arg['text'].lower()) or ('nymex' in arg['text'].lower()) or ('U.S. oil' in arg['text']):
                WTI = True
                break
            elif (arg['role'] == 'Attribute') and (arg['text'] == 'prices' or arg['text'] == 'price'):
                generic_price = True
    
        if WTI:
            price_details = retrieve_price_details(event['arguments'])       
            return True, 'WTI', movement_label[0], price_details 
        elif Brent:
            price_details = retrieve_price_details(event['arguments'])
            return True, 'Brent', movement_label[0], price_details         
        elif generic_price:
            return True, '', movement_label[0], {}
        else:
            return False, '', '', {}
    
    else:
        return False, '', '', {}

def retrieve_price_details(args):
    '''
    Purpose: retrieve price details
    '''
    ## Interested roles: Difference, Final_value, Initial_value
    flat_args = {}
    flat_args['Difference'] = 0
    flat_args['Difference_money'] = 0
    flat_args['Initial_value'] = 0
    flat_args['Final_value'] = 0
    for arg in args:
        ### Final Value ##
        if arg['role'] == 'Final_value':
            flat_args['Final_value'] = process_numbers_in_text(arg['text'])
        elif arg['role'] == 'Initial_value':
            flat_args['Initial_value'] = process_numbers_in_text(arg['text'])
        elif (arg['role'] == 'Difference') and (arg['entity-type'] == "Percentage"):
            flat_args['Difference'] = process_numbers_in_text(arg['text'])
        elif (arg['role'] == 'Difference') and (arg['entity-type'] == "Money" or arg['entity-type'] == 'Price_unit'):
            flat_args['Difference_money'] = process_numbers_in_text(arg['text'])
        
    if flat_args['Difference'] == 0:
        if flat_args['Difference_money'] != 0 and flat_args['Initial_value'] != 0:   ### do calculation
            flat_args['Difference'] = flat_args['Difference_money'] / flat_args['Initial_value']
        elif flat_args['Difference_money'] != 0 and flat_args['Final_value'] != 0:   ### do calculation
            flat_args['Difference'] = flat_args['Difference_money'] / abs(flat_args['Final_value'] - flat_args['Difference_money']) * 100
        elif flat_args['Difference_money'] == 0 and flat_args['Initial_value'] != 0 and flat_args['Final_value'] != 0:
            flat_args['Difference'] = abs(flat_args['Final_value'] - flat_args['Initial_value']) / flat_args['Initial_value'] * 100
        
    return flat_args
        
    
def restructure_data(data_list, file_name):
    summary= {}
    summary['file'] = file_name
    summary['header'] = ''
    summary['body'] = []
    for entry in data_list:
        
        ######    Header ###########
        if entry['sent_type'] == 'H' and entry['price'] == 'n':
            summary['header'] = entry['event_type'] + ': ' + entry['trigger']['text'] + \
              ', ' + entry['polarity'] + ', ' + entry['modality'] + ', ' + entry['intensity'] + ', ' + entry['arguments']
               
        
        ######     Sentence - Non-price events   ###### 
        if entry['sent_type'] == 'S' and entry['price'] == 'n':
            summary['body'].append(entry['event_type'] + ': ' + entry['trigger']['text'] + ', ' + entry['polarity'] + ', ' + entry['modality'] + ', ' + entry['intensity'] + ', ' + entry['arguments'])
        
        ######  Price events   ###### 
        if entry['sent_type'] == 'S' and entry['price'] == 'y' and entry['commodity_type'] == 'Brent':
            summary['brent_label'] = entry['label']
            summary['brent_difference'] = entry['arguments']['Difference']
        elif entry['sent_type'] == 'S' and entry['price'] == 'y' and entry['commodity_type'] == 'WTI':
            summary['wti_label'] = entry['label']
            summary['wti_difference'] = entry['arguments']['Difference']
        
    return summary
    
            
def flatten_arguments(arguments):
    '''
    input:  [{'start': 33, 'end': 34, 'role': 'Attribute', 'entity-type': 'Financial_attribute', 'text': 'prices', 'standoff_id': 104}, 
                {'start': 32, 'end': 33, 'role': 'Item', 'entity-type': 'Commodity', 'text': 'Oil', 'standoff_id': 103}]
    output: 'Attribute: prices, Item: Oil' (a string of arguments in form of Role: Text)
    {'Difference': 0, 'Difference_money': 0, 'Initial_value': 0, 'Final_value': 110.0}
    '''
    flat_arg = ''
    for arg in arguments:
        flat_arg += arg['role'] + ': ' + arg['text'] + ', '
    
    return flat_arg[:-2]       
    
    
def process_numbers_in_text(text):
    '''
    difference: percentage (5, %)
                money (cents, $, USD, A$, $ in million)
                price unit ($x.xx per barrel, /bbl. cents per barrel)
                
    initial value / final_value: money (cents, $, USD, A$, $ in million)
                   price unit ($x.xx per barrel, /bbl, cents per barrel)
    '''
    value = 0
    
    
    ####    MONEY in USD  ####
    if len([x for x in re.findall('cents', text)]) > 0:    
        value = float(re.findall('\d{1,3}(?:,\d{3})*(?:\.\d+)?(?=\s)', text)[0])/100
        
    if len([x for x in re.findall('USD', text)]) > 0:
        value = float(re.findall('(?<=USD)\d{1,3}(?:,\d{3})*(?:\.\d+)?', text)[0].replace(',',''))
        
    if len([x for x in re.findall('\$', text)]) > 0:
        value = float(re.findall('(?<=\$)\d{1,3}(?:,\d{3})*(?:\.\d+)?', text)[0].replace(',',''))
        

    ####    PERCENTAGE  ####
    
    if len([x for x in re.findall('\%', text)]) > 0:
        value = float(re.findall('\d{1,3}(?:\.\d+)*%', text)[0].replace('%',''))
        
    if len([x for x in re.findall('percent', text)]) > 0:
        value = float(re.findall('\d{1,3}(?:\.\d+)*', text)[0].replace('%',''))
        
    if len([x for x in re.findall('per cent', text)]) > 0:
        value = float(re.findall('\d{1,3}(?:\.\d+)*', text)[0].replace('%','')) 
        
    return value
        
def write_to_json(output_file, data):
    # convert into json
    with open(output_file, 'w', encoding="utf8") as f:
        json.dump(data, f, indent=2)
        

def main():
    DATA_DIRECTORY = "raw data"
    OUTPUT_DIRECTORY = "dataset"
    
    data_files = [f for f in listdir(DATA_DIRECTORY) if isfile(join(DATA_DIRECTORY, f)) and f.split('.')[-1] == 'json']

    json_data = []
    
    for file_name in data_files:  
        try:
            with open(join(DATA_DIRECTORY, file_name), 'r') as f:
                data = json.load(f)
                process_file(data, file_name)
                
        except Exception as e:
            print(e)
    write_to_json(join(OUTPUT_DIRECTORY, 'processed_data.json'), json_data)

if __name__ == "__main__":
    main()