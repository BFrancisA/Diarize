# Diarize
Diarization Project

- Proposal:  Proposal.pdf
- Project Report: ML-Project-Report.pdf
 
All development was carried out using:
- PyCharm 2018.2 (Community Edition)
- Build #PC-182.3684.100, built on July 24, 2018
- JRE: 1.8.0_152-release-1248-b8 amd64
- JVM: OpenJDK 64-Bit Server VM by JetBrains s.r.o
- Windows 10 10.0 

The easiest way to run the python functions in this project it to load 
the project into PyCharm.  All the dependencies will be loaded by PyCharm
using Conda.

The programs developed to convert the audio data files to 
mel-frequency cepstrums and build the superframes is prior 
work developed in C++.  This software is not included.  

The SWBCell2 and SWBPhase3 datasets are copyrighted and 
cannot be distributed, however, a small dataset of derived 
superframe data was included in the Diarize project on github.

Samples of training data are supplied in the directory: 

- **Data-small**:   This contains example data of 5 speakers 
taken from the SWBCell2 dataset.  Each file contains 
one superframe.  The file naming convention follows 
the familiar format of file-number.label  In total there are 
1,250 superframes.


Diarization test data is supplied in the directory: 
- **RNNTestData-Trim-Small**:  This small dataset contains 
5 full length audio recordings converted to superframes. Each 
superframe file is named using the format 
frame-number.speaker-pin-number

## Training Functions
- Train-cnn.py :		Used to train the 2-D CNN model.
- Train-R-CNN.py :	Used to train the R-CNN model.

The above training functions can be run in their current form
to train the model using the dataset **Data-small** (5 speakers, 1250 
superframes).  The actual models that were created and evaluated were 
trained using 65,000 superframes with 260 speakers.

Note: the design for the R-CNN followed in part the implementation 
of an R-CNN given in:  
https://github.com/sharathadavanne/multichannel-sed-crnn/blob/master/sed.py

In the above reference they create a similar R-CNN model to be used
for bird audio detection. This reference is used as a starting point 
for this model design.  

## Diarization Function

- Diarize-Model-Clustering.py  
  This function runs diarization on recordings that have been preprocessed into
  superframes.  This function can be run as is to diarize the  5 recordings 
  conntained in the directory **RNNTestData-Trim-Small**.
    

*Diarize-Model-Clustering.py*  produces a diarization accuracy 
score measuring the percentage of superframes correctly 
diarized as either speaker 1 or speaker 2 in these two-speaker 
recordings.  Double-talk frames are marked with PIN 9999.  These are not 
included in the accuracy calculations. 

