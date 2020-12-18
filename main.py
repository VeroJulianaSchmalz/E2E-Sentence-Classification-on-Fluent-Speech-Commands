# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:56:04 2020

@author: iv3r0
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
parser.add_argument('-m', '--model', type = str, help = "model name", default='best_model.pkl')
parser.add_argument('-b', '--blocks', type = int, help='blocks',default = 5)
parser.add_argument('-r', '--repeats', type = int, help='repeats', default = 2)
parser.add_argument('-lr', '--learning_rate', type = float, help = 'learning rate', default = 0.001)
parser.add_argument('-e', '--epochs', type = int, help = 'epochs', default = 100)
parser.add_argument('-w', '--workers', type = int, help='workers',default = 1)
parser.add_argument('-p', '--pathdataset', type = str, help='pathdataset')
parser.add_argument('--batch_size', type = int, help='batch_size',default = 200)
parser.add_argument('--n_classes', type = int, help='number of output classes',default = 248)

arg = parser.parse_args()
path_dataset = arg.pathdataset
numworkers = arg.workers
tcnBlocks = arg.blocks
tcnRepeats = arg.repeats
learning_rate = arg.learning_rate
epochs = arg.epochs
modelname = arg.model
batch_size = arg.batch_size
n_classes=arg.n_classes
##Set device as cuda if available, otherwise cpu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

train_data = fsc_data( path_dataset + '/data/train_data.csv', max_len = 64000)
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': numworkers}
train_set_generator=data.DataLoader(train_data,**params)

valid_data = fsc_data(path_dataset + '/data/valid_data.csv',max_len = 64000)
params = {'batch_size': batch_size,
          'shuffle': False,
          'num_workers': numworkers}
valid_set_generator=data.DataLoader(valid_data,**params)

model = TCN(n_blocks=tcnBlocks, n_repeats=tcnRepeats, out_chan=n_classes).to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

criterion = torch.nn.CrossEntropyLoss()

best_accuracy = 0

for e in range(epochs):
    for i, d in enumerate(train_set_generator):
        model.train()
        f, l = d
        y = model(f.float().to(device))
        loss = criterion(y,l.to(device))

        print("Iteration %d in epoch%d--> loss = %f"%(i,e,loss.item()),end='\r')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i%20 == 0:            
            model.eval()
            correct = []
            for j,eval_data in enumerate(valid_set_generator):
                feat,label = eval_data

                y_eval = model(feat.float().to(device))
                _, pred = torch.max(y_eval.detach().cpu(),dim=1)

                correct.append((pred == label).float())
                if j > 10:
                    break
            acc = (np.mean(np.stack(correct)))
            iter_acc = 'iteration %d epoch %d--> %f (%f)'%(i, e, acc, best_accuracy)  #accuracy
            print(iter_acc)   
            
       
            if acc > best_accuracy:
                improved_accuracy = 'Current accuracy = %f (%f), updating best model'%(acc,best_accuracy)
                print(improved_accuracy)
                best_accuracy = acc
                best_epoch = e
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), modelname)
                

