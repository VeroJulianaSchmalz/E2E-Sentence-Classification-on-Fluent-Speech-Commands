# End-to-End Sentence-Classification-FSCs-Schmalz
End-to-End Sentence Classification using the Fluent Speech Commands Dataset 

Using the opensource Fluent Speech Command dataset (available at [fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research]), we consider an **End-2-End phrase classification** task. We adopt a neural framework that classifies the **30,043 utterances** into the **248** possible **sentences** contained in the dataset. 

The network receives as input **fixed length** sequences of **40 Mel filter-banks**. The signal length is limited to 4 seconds or **64000** samples. Filter banks are computed on **20ms window** with **10ms hop size**, resulting in **400** frames. Three models are used to experiment the execution of the task: TCN, CNN, honk.  
