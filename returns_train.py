
import logging
import argparse
import torch
from torch.nn import init
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from datetime import datetime
from torch.utils import data

from utils.data_loader import CommodityPriceDataset
from model.CommodityPricePrediction import CommodityPrice
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from utils.const import POLARITY, MODALITY, INTENSITY, TRIGGERS, PRICE
all_price = PRICE

datetime_object = str(datetime.now())
date = datetime_object.split(' ')[0]
time = datetime_object.split(' ')[1].replace(":", "_")

filename = 'runs/logfiles/returns/output_' + date + time + '.log'

### Setup logging
logging.basicConfig(level=logging.INFO, filename=filename, filemode='w')
logger = logging.getLogger(__name__)

def train_regressionReturns(model, train_dataset, optimizer, criterion, device, args):
    tr_loss = 0
    model.train()
    total_len = len(train_dataset)

    for i, batch in enumerate(train_dataset):
        header, body, movement_label, WTI_difference, Brent_difference = batch
        batch_size = len(movement_label)

        WTI_difference = WTI_difference.type(torch.FloatTensor)

        # Move input to GPU
        header_input = header['input_ids'].squeeze(1).to(device)
        body_input = body['input_ids'].squeeze(1).to(device)
        WTI_difference = WTI_difference.to(device)
        Brent_difference = Brent_difference.to(device)
        movement_label = movement_label.to(device)

        ## Ablation Study by varying different input scope
        if args.news_part == "header":
            regression_input = header_input
        elif args.news_part == "all":
            regression_input = torch.cat((header_input, body_input), 1)   
        else:
            regression_input = body_input

        optimizer.zero_grad()
        price_hat = model.regression_price(regression_input)  
        
        
        ##########   MSE loss   ################        
        loss = criterion(price_hat, WTI_difference.unsqueeze(1))    ## torch.sqrt

        tr_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    return (tr_loss / total_len)     

def eval_regressionReturns(model, train_dataset, optimizer, criterion, device, args):
    tr_loss = 0
    MSE, MAPE = 0, 0
    model.eval()
    total_len = len(train_dataset)
    price_hat_all, WTI_difference_all = [], []

    with torch.no_grad():
        for i, batch in enumerate(train_dataset):
            header, body, movement_label, WTI_difference, Brent_difference = batch
            batch_size = len(movement_label)

            WTI_difference = WTI_difference.type(torch.FloatTensor)

            # Move input to GPU
            header_input = header['input_ids'].squeeze(1).to(device)
            body_input = body['input_ids'].squeeze(1).to(device)
            WTI_difference = WTI_difference.to(device)
            Brent_difference = Brent_difference.to(device)
            movement_label = movement_label.to(device)

            ## Ablation Study by varying different input scope
            if args.news_part == "header":
                regression_input = header_input
            elif args.news_part == "all":
                regression_input = torch.cat((header_input, body_input), 1)   
            else:
                regression_input = body_input

            price_hat = model.regression_price(regression_input)            

            ##########   MSE loss   ################        
            loss = criterion(price_hat, WTI_difference.unsqueeze(1))            
            
            tr_loss += loss.item()
                    
            price_hat_all += price_hat.cpu().numpy().tolist()  
            WTI_difference_all += WTI_difference.unsqueeze(1).cpu().numpy().tolist()
        
    MSE = mean_squared_error(WTI_difference_all, price_hat_all, squared=False)
    MAPE = mean_absolute_percentage_error(WTI_difference_all, price_hat_all)
    return (tr_loss / total_len), MSE, MAPE
        
        
def main():
    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--train_data_file", default="dataset/train_data.json", type=str, help="The input training data file (a json file).")    #required=True,
    parser.add_argument("--eval_data_file", default="dataset/eval_data.json", type=str, help="The input testing data file (a json file).")
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size per GPU for training.")
    parser.add_argument("--learning_rate", default= 0.00002, type=float, help="The learning rate for model's optimizer.")
    parser.add_argument("--n_epochs", default=1, type=int, help="Number of training epochs.")
    parser.add_argument("--dropout", default=0.25, type=float, help="dropout rate")
    parser.add_argument("--content_scope", default="all", type=str, help="content scope - all, minus_arguments_properties, minus_arguments, minus__properties")
    parser.add_argument("--news_part", default="all", type=str, help="which part of the news articles - all, header, body")
    parser.add_argument("--output_dir", default="checkpoints", type=str, help="Output directory to write results to")

    args = parser.parse_args()
    
    #logger.info("Training/evaluation parameters %s", args)          
        
    ### Define Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    model_regression = CommodityPrice(args, len(all_price), model)
    model_regression = model_regression.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)    ## use back the same optimizer
    
    
    ### training
    trainset = args.train_data_file
    train_dataset = CommodityPriceDataset(trainset, args.content_scope)
    train_dataset = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle=False)
    
            
    ### Evaluation    
    testset = args.eval_data_file
    test_dataset = CommodityPriceDataset(testset, args.content_scope)
    test_dataset = data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)
    
    global_step = 0
    print('=====Commodity Price Movement Classification=====')
    logger.info("=====Commodity Price Movement Classification=====")
    logger.info("***** Running training *****")
    for epoch in range(1, args.n_epochs + 1):
        
        global_step += 1
        print('epoch # ' + str(epoch))
        tr_loss = train_regressionReturns(model_regression, train_dataset, optimizer, criterion, device, args)
        print('training loss : ' + str(tr_loss))
        
        eval_loss, MSE, MAPE = eval_regressionReturns(model_regression, test_dataset, optimizer, criterion, device, args)
        print('evaluation loss : ' + str(eval_loss))       
        print('MSE - ' + str(MSE))
        print('MAPE -' + str(MAPE))
        
        ## Write to logging file
        logger.info('training loss : ' + str(tr_loss))
        logger.info('epoch # ' + str(epoch))
        logger.info('f1 - ' + str(MSE))
        logger.info('MCC -' + str(MAPE))          


if __name__ == "__main__":
    main()