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
from models import CNNNet, honk, TCN
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
#parser.add_argument('-n', '--net', type = str, help = 'net', choices = ['CNNNet','honk', 'TCN'], default='CNNNet')
parser.add_argument('-b', '--blocks', type = int, help = 'blocks')
parser.add_argument('-r', '--repeats', type = int, help='repeats')
parser.add_argument('-w', '--workers', type = int, help='workers')


#storing params 
arg = parser.parse_args()
model_name = arg.model



test_data = fsc_data('fluent_speech_commands_dataset/data/test_data.csv',max_len = 64000)
params = {'batch_size': 20,      #n returned phrases 
              'shuffle': False,
               'num_workers': arg.workers} 
test_set_generator=data.DataLoader(test_data,**params)



valid_data = fsc_data('fluent_speech_commands_dataset/data/valid_data.csv',max_len = 64000)
params = {'batch_size': 20,    
              'shuffle': False,
               'num_workers': arg.workers} 
valid_set_generator=data.DataLoader(valid_data,**params)


model = TCN(n_blocks=arg.blocks,n_repeats=arg.repeats)


#loading the model 
model.load_state_dict(torch.load(model_name))
model.eval()
model.cuda()


correct_test = []

for i, data in enumerate(test_set_generator):
    print(i,end = '\r')
    feat,label=data

    z_eval = model(feat.float().cuda())                
    _, pred_test = torch.max(z_eval.detach().cpu(),dim=1)
    correct_test.append((pred_test==label).float())


acc_test= (np.mean(np.hstack(correct_test)))  
print("The accuracy on test set is %f" %(acc_test))


test_val=[]
for i, data in enumerate(valid_set_generator):

    feat,label=data

    a_eval = model(feat.float().cuda())                
    _, pred_test = torch.max(a_eval.detach().cpu(),dim=1)
    test_val.append((pred_test==label).float())


acc_val= (np.mean(np.hstack(test_val)))  
print("The accuracy on the validation set is %f" %(acc_val))
