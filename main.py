import torch
import argparse
import os, sys, json
from datetime import datetime
from data import get_dataloaders
from engine import *

parser = argparse.ArgumentParser()

parser.add_argument('--device_id', default=0, type=int,
                    help='the id of the gpu to use')    
# Model Related
parser.add_argument('--model', default='baseline', type=str,
                    help='Model being used')
parser.add_argument('--pt_ft', default=1, type=int,
                    help='Determine if the model is for partial fine-tune mode')

# Data Related
parser.add_argument('--bz', default=32, type=int,
                    help='batch size')
parser.add_argument('--shuffle_data', default=1, type=int,
                    help='Shuffle the data')
parser.add_argument('--normalization_mean', default=(0.485, 0.456, 0.406), type=tuple,
                    help='Mean value of z-scoring normalization for each channel in image')
parser.add_argument('--normalization_std', default=(0.229, 0.224, 0.225), type=tuple,
                    help='Mean value of z-scoring standard deviation for each channel in image')


# Other Choices & hyperparameters
parser.add_argument('--epoch', default=25, type=int,
                    help='number of epochs')
    # for loss
parser.add_argument('--criterion', default='cross_entropy', type=str,
                    help='which loss function to use')
    # for optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    help='which optimizer to use')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=0, type=float,
                    help='weight decay') # TODO: Try with 1e-4
# for scheduler
parser.add_argument('--lr_scheduling', default=0, type=int,
                    help='Enable learning rate scheduling')
parser.add_argument('--lr_scheduler', default='steplr', type=str,
                    help='learning rate scheduler') 
parser.add_argument('--step_size', default=7, type=int,
                    help='Period of learning rate decay')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Multiplicative factor of learning rate decay')

args = vars(parser.parse_args())

def main(args):
    device = torch.device("cuda:{}".format(args['device_id']) if torch.cuda.is_available() else 'cpu')
    dataloaders = get_dataloaders('food-101/train.csv', 'food-101/test.csv', args)
    model, criterion, optimizer, lr_scheduler = prepare_model(device, args)
    
    model = train_model(model, criterion, optimizer, lr_scheduler, device, dataloaders, args)

if __name__ == '__main__':
    main(args)
