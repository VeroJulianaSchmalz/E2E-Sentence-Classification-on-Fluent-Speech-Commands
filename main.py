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
parser.add_argument('-m', '--model', type=str, help="model name", required=True)
parser.add_argument('-n', '--net', type=str, help='net', choices=['CNNNet','honk', 'TCN'], default = 'CNNNet')
parser.add_argument('-b', '--blocks', type=int, help='blocks')
parser.add_argument('-r', '--repeats', type=int, help='repeats')
parser.add_argument('-lr', '--learning_rate', type=int, help='learning rate', default = 0.001)        #originally 0.01 
parser.add_argument('-e', '--epochs', type=int, help=' epochs', default = 100)                        #originally 30

#storing params 
arg = parser.parse_args()

test_data = fsc_data('fluent_speech_commands_dataset/data/test_data.csv',max_len=64000)
params = {'batch_size': 20,   
              'shuffle': False}  
test_set_generator=data.DataLoader(test_data,**params)
    
  
train_data = fsc_data('fluent_speech_commands_dataset/data/train_data.csv',max_len=64000)
params = {'batch_size': 200,
              'shuffle': True,
              'num_workers': 6}
train_set_generator=data.DataLoader(train_data,**params)
  

valid_data = fsc_data('fluent_speech_commands_dataset/data/valid_data.csv',max_len=64000)
params = {'batch_size': 20,   
              'shuffle': False,
              'num_workers': 6} 
valid_set_generator=data.DataLoader(valid_data,**params)



model= TCN(n_blocks=arg.blocks,n_repeats=arg.repeats).cuda()                                   #original param values 5-2(changed params for the experiments) 
#model= CNNNet(n_frames=401, n_feats=40, kernel=5, max_pooling=2).cuda()
#model= honk(width=401).cuda()
    
optimizer= optim.Adam(model.parameters(), lr=arg.learning_rate)                          #lr for the experiments=0.001
###changed lr for TCN experiments, originally 0.01
criterion= torch.nn.CrossEntropyLoss()

best_accuracy = 0
epochs = arg.epochs                                                                #originally for each model 30 epochs
for e in range(epochs):
    for i, data in enumerate(train_set_generator):
        model.train()
        f,l = data
     
        y= model(f.float().cuda())

        loss= criterion(y,l.cuda())
        print("Iteration %d in epoch%d--> loss = %f"%(i,e,loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i%20 == 0:            
            model.eval()
            correct = []
            for j,eval_data in enumerate(valid_set_generator):
                feat,label = eval_data

                y_eval = model(feat.float().cuda())                
                _, pred = torch.max(y_eval.detach().cpu(),dim=1)

                correct.append((pred == label).float())
                if j > 10:
                    break
            acc= (np.mean(np.stack(correct)))
            iter_acc= 'iteration %d epoch %d--> %f'%(i,e,acc)  #accuracy 
            print(iter_acc)   
            
       
            if acc> best_accuracy:
                improved_accuracy = 'Current accuracy = %f (%f), updating best model'%(acc,best_accuracy)
                print(improved_accuracy)
                best_accuracy = acc
                best_epoch= e
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), arg.model)         #change name for each model
            else:
                if e-best_epoch>5: 
                    break 
                
