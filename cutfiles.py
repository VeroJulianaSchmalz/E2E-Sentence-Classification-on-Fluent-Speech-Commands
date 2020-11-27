# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:48:01 2020

@author: iv3r0
"""



import soundfile as sf
import numpy
import glob   
import os 
import matplotlib.pyplot as plot
import librosa 

f_files=[]
fcut_files=[]

for i, file in enumerate(glob.glob('fluent_speech_commands_dataset/wavs/speakers/**/*.wav')):
        
        print(i,end='\r')
        f,sr = sf.read(file) 
        f_files.append(len(f))
        
        #plot.plot(f)
        #plot.show()
        #print(file,i)
        dir_name = os.path.dirname(file)
        dir_name_cut = dir_name.replace('wavs','wavscut') 
        
        os.makedirs(dir_name_cut, exist_ok=True)
        filename = os.path.basename(file)
        
        fcut,index= librosa.effects.trim(f,frame_length=2098, hop_length=562)  ### DEFAULT  frame_length=2048, hop_length=512
    
        fcut_files.append(len(fcut))
      
        sf.write(dir_name_cut+'/'+ filename, fcut, sr)                         ##creation of filename in new folder 
        

### plotting histogram
plot.hist(f_files, bins='auto')  
plot.xlim(0, 100000)
plot.title("Histogram full wav files")
plot.show()       

plot.hist(fcut_files, bins='auto')  
plot.title("Histogram cut wav files")
plot.xlim(0, 100000)
plot.show()       