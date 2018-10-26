# Diarize
Diarization Project

 
All development was carried out using:
- PyCharm 2018.2 (Community Edition)
- Build #PC-182.3684.100, built on July 24, 2018
- JRE: 1.8.0_152-release-1248-b8 amd64
- JVM: OpenJDK 64-Bit Server VM by JetBrains s.r.o
- Windows 10 10.0 

The programs developed to convert the audio data files to 
mel-frequency cepstrums and build the superframes is prior 
work developed in C++.  This software is not included.  

The SWBCell2 and SWBPhase3 datasets are copyrighted and 
cannot be distributed, however, small dataset of derived 
superframe data was included in the Diarize project on github.

Samples of training data are supplied in the directory:  
Data-small.  This contains example data of 5 speakers 
taken from the SWBCell2 dataset.  Each file contains 
one superframe.  The file naming convention follows 
the familiar format of file-number.label


Diarization test data is supplied in the directory: 
RNNTestData-Trim-Small.  This small dataset contains 
5 full length audio recordings converted to superframes.  
Each superframe file is named using the format 
frame-number.speaker-pin-number

## Training Functions
Train-cnn.py		Used to train the 2-D CNN model.
Train-R-CNN.py	Used to train the R-CNN model.

## Diarization Function

Diarize-Model-Clustering.py  :  Used to diarize recordings 
represented as superframes.  Produces a diarization accuracy 
score measuring the percentage of superframes correctly 
diarized as either speaker 1 or speaker 2 in these two-speaker 
recordings.

