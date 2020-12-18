# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:36:18 2020

@author: iv3r0
"""

import torch
from torch.utils import data
import soundfile as sf
import numpy as np
from scipy import signal  
import librosa 


class fsc_data(data.Dataset):
    def __init__(self, csvfilename, max_len=64000, win_len=0.02):
        self.max_len = max_len
        self.audioid = []
        self.transcriptions = []
        self.intent = []
        self.win_len = win_len
        self.eps = np.finfo(np.float64).eps
        
        with open(csvfilename) as fcsv:
            lines = fcsv.readlines()
            #print(lines) 
            for l in lines[1:]:          
                items = l[:-1].split(',')
                self.audioid.append(items[1])
                if len(items) == 7:
                    self.transcriptions.append(items[3])
                else:
                    self.transcriptions.append((" ").join(items[3:5]))

                self.intent.append(tuple(items[-3:]))
                #exit()
        utteranceset = sorted(list(set(self.transcriptions)))
        self.sentence_labels = [utteranceset.index(t) for t in self.transcriptions]
        intentset = sorted(list(set(self.intent)))
        self.intent_labels = [intentset.index(t) for t in self.intent]

    def __len__(self):
        return len(self.audioid)
                
    def __getitem__(self, index):
        audiofile = self.audioid[index]
        #print(audiofile)
        f,sr = sf.read('fluent_speech_commands_dataset/'+audiofile)              
        f = f [:self.max_len] 
        n_fft = int(self.win_len*sr)
        if len(f) < self.max_len: 
            ff = np.pad(f, [(0, self.max_len - f.shape[0]),], mode='constant')
            f=ff
        label = self.sentence_labels[index]
        #label = self.intent_labels[index]

        # extracting Mel filters
        filters = librosa.filters.mel(sr,n_fft,n_mels=40)
        window  = signal.hamming(n_fft, sym=False)
        spectrogram = np.abs(librosa.stft(y=f+self.eps ,n_fft=n_fft, win_length=n_fft, hop_length=n_fft//2, center=True,window=window))
        melspectrum = np.log(np.dot(filters, spectrogram) + self.eps)
         
        return torch.from_numpy(melspectrum), label         


if __name__ == "__main__":
    mydata = fsc_data('fluent_speech_commands_dataset/data/test_data.csv',  max_len=64000)
    print(mydata)
    print(mydata.__len__())

    params = {'batch_size': 5,   
              'shuffle': False} 
    test_set_generator=data.DataLoader(mydata,**params)
    
    for x,y in test_set_generator:
        print(y)      
        print(x.shape)  
        break 