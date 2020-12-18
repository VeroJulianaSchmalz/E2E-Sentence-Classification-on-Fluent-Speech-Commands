#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:49:54 2020

@author: vschmalz
"""


import copy
import torch
from torch.utils import data
import soundfile as sf
import numpy as np
from scipy import signal  
import librosa 
from models import TCN
from dataset_fbank import fsc_data
import torch.optim as optim 
import torch.nn
from torch.autograd import Variable
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#reading params
parser = argparse.ArgumentParser(description='Desciption')
parser.add_argument('-m', '--model', type = str, help = "model name", required=True)
parser.add_argument('-b', '--blocks', type = int, help = 'blocks')
parser.add_argument('-r', '--repeats', type = int, help='repeats')
parser.add_argument('-w', '--workers', type = int, help='workers',default=2)
parser.add_argument('-p', '--pathdataset', type = str, help='pathdataset')
parser.add_argument('--batch_size', type = int, help='pathdataset',default = 100)

#storing params 
arg = parser.parse_args()
model_name = arg.model
path_dataset= arg.pathdataset
batch_size=arg.batch_size

test_data = fsc_data(path_dataset + '/data/test_data.csv',max_len = 64000)
params = {'batch_size': batch_size,'shuffle': False,'num_workers': arg.workers}
test_set_generator=data.DataLoader(test_data,**params)

valid_data = fsc_data(path_dataset + '/data/valid_data.csv',max_len = 64000)
params = {'batch_size': batch_size,'shuffle': False, 'num_workers': arg.workers}
valid_set_generator=data.DataLoader(valid_data,**params)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

model = TCN(n_blocks=arg.blocks,n_repeats=arg.repeats,out_chan=31)

#loading the model 
model.load_state_dict(torch.load(model_name))
model.eval()
model.to(device)

correct_test = []
for i, d in enumerate(test_set_generator):

    feat,label=d
    print('Iter %d (%d/%d)'%(i,i*batch_size,len(test_data)),end='\r')
    z_eval = model(feat.float().to(device))
    _, pred_test = torch.max(z_eval.detach().cpu(),dim=1)
    correct_test.append((pred_test == label).float())


acc_test= (np.mean(np.hstack(correct_test)))  
print("The accuracy on test set is %f" %(acc_test))


correct_valid=[]
for i, d in enumerate(valid_set_generator):
    print('Iter %d (%d/%d)'%(i,i*batch_size,len(valid_data)),end='\r')
    feat,label=d

    a_eval = model(feat.float().to(device))
    _, pred_test = torch.max(a_eval.detach().cpu(),dim=1)
    correct_valid.append((pred_test == label).float())


acc_val= (np.mean(np.hstack(correct_valid)))
print("The accuracy on the validation set is %f" %(acc_val))
