# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 19:00:05 2020

@author: iv3r0
"""

### This script finds all wav files in wavs/speakers and computes the maximum file length in samples
import soundfile as sf
import numpy
import glob   
len_lin = []
names_list = []
for file in glob.glob('wavs/speakers/**/*.wav'): ##finds all files
    f,sr = sf.read(file) 
    ##load the audio samples in f
    names_list.append(file)
    len_lin.append(len(f)) 
    
#print(len(len_lin))    #30043
len_lin = numpy.array(len_lin)

max = numpy.amax(len_lin)

id_max = numpy.where(len_lin==numpy.amax(len_lin))
# print(id_max)
# print('Returned tuple of arrays :', id_max)
# print('List of Indices of maximum element :', id_max[0])
# print(len_lin.shape)
# (len_lin[id_max], file)
print('The maximum file length in the dataset is %d samples in file %s'%(len_lin[25885],names_list[25885]))

######### if separated error of not defined len_lin #########

import matplotlib.pyplot as plt

plt.hist(len_lin, bins='auto')  # arguments passed to np.histogram.
plt.title("Histogram dataset full-lenght wav files ")
plt.show()