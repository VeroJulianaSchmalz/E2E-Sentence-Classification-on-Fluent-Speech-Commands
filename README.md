# End-to-End Sentence-Classification-FSCs-Schmalz

End-to-End Sentence Classification based on the Fluent Speech Commands Dataset 

Using the opensource Fluent Speech Command dataset (available at https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research), we consider an **End-2-End phrase classification** task. The dataset contains **30,043 utterances** and **248** possible **sentences**. The utterances are pronounced by both native and non-native English speakers and include phrases like "turn off the lights in the kitchen" or "heat up in the living room", while the possible sentences are intended to define the action, object and location of them, for example "deactivate, lights, kitchen" or "increase, heat, living room" for the previously mentioned phrases. 

In order to address the task we adopt a neural framework that classifies the utterances to the possible sentences. 
The model used in the experiment is a Time Convoluted Network (**TCN**, available at https://github.com/asteroid-team/asteroid from https://github.com/popcornell/OSDC. The network receives as input **fixed length** sequences of **40 Mel filter-banks**. The signal length is limited to 4 seconds or **64000** samples. Filter banks are computed on **20ms window** with **10ms hop size**, resulting in **400** frames. 

## Pre-processing

- *cutfiles.py*:
```
fcut,index= librosa.effects.trim(f,frame_length=2098, hop_length=562)
```

*./fluent_speech_commands_dataset/* identifies the folder in which the data are to be found.  


## Training 

- *main.py* :  
```
python3.6 main.py -n TCN -m models/tcn_b5r2.pkl -b 5 -r 2 -lr 0.001 -e 100
```

 * *n*: type of net 
 * *m*: model
 * *b*: number of blocks of the TCN network 
 * *r*: number of repeats of the TCN network 
 * *lr*: learning rate 
 * *e*: number of epochs 
 
 
### Evaluation 

- *evaluation.py*:  
```
python3.6 evaluation.py -n TCN -m models/tcn_b5r2.pkl -b 5 -r 2 
```

Using the training parameters suggested above, the obtained results should be *0.816870* accuracy on the validation set and *0.934880* accuracy on the evaluation set. 
