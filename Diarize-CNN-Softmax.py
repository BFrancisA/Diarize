"""
Run Diarization using the CNN only for Speaker Classification
This method tries to detect speaker changes using cosine distance between successive
superframes.  It does not work well.  Instead the Gaussian Weighted Clustering is much
more effective and is the preferred approach.  See:  Diarize-Model-Clustering.py
"""

import os
import sys
import numpy as np
from scipy import spatial

import GetSuperframesInfo
import LoadDataset
import CnnModel_2
import GetSpeakerLabels
import Clustering


"""
Load the test data for evaluating the diaraization.  This is a set of recordings, where each recording 
has already been converted to superframe features.  Each directory holds one audio recording of a two-person 
conversation converted to a time ordered sequence of super frames.  Each frame is named <superframe index>.<pin>
pin = p.i.n. =  personal identification number.
 
For example:

RecordingData 
          /sw_45001               <- the recording dir
               /0001.6300         <- 1st superframe binary file.  Here speaker 6300 is speaking.
               /0002.6300         <- 2nd superframe and 6300 is still speaking.
   
Here 0001.6300 is name of the first superframe in the recording.  The speaker PIN number is 6300.
For the RNN, we convert the speaker PINs to just speaker 1, speaker2,... We just care about speaker changes.
Usually there are only 2 speakers per recording.
We reserve 0 for the start of the sequence
Speaker i is the i'th speaker
Speaker 9999 is reserved for a double-talk superframe where both speakers are talking at one time.

We need an array of superframes and an array of labels.  We have one label for each superframe in the sequence.
   
"""

"""
Use the CNN to classify the each superframe as either new speaker or same speaker.
To do this, first calculate the speaker embeddings for each superframe in the recording.
Then apply normalization over all speaker embeddings.  This is done so that we can use a fixed threshold 
to determine a speaker change based on cosine similarity.
Next run cosine similarity to detect speaker changes.
Record the change as a sequence of labels: 1 for speaker 1, 2 for speaker 2, ... 
to mark end of recording.

To evaluate the accuracy:  
    For each superframe compare the assigned speaker with the true speaker for the superframe.
    Ignore the result for superframes that have doubletalk. 
     
"""

doubleTalkLabel = 9999
# if dist between two consecutive features is greater than this value, then a new speaker starts.
newSpeakerThreshold = 0.6

# If dist between a feature and a last seen feature for a last seen speaker is less than this value
# then this is a repeat of the previously seen speaker.
sameSpeakerThreshold = 0.6

# Parent directory containing all the recording.  There is one recording per subdirectory.
all_files_path = 'RNNTestData-Trim-Small'
#all_files_path = 'C:\Diarization\RNNTrainingData-Trim'
#all_files_path = 'C:\Diarization\RecordingDataTest'

# Get a list of the recording directories.
recordings = [name for name in os.listdir(all_files_path)]
recordingsDirs = [os.path.join(all_files_path, fullPath) for fullPath in recordings]
print(recordingsDirs)

trueSpeakerCount = 2
number_of_targets = 260   # total number of speakers
dropout_rate = 0  # not used here
tensor_shape = (1, 64, 32, 1)

model = CnnModel_2.create_model(number_of_targets, tensor_shape, dropout_rate)
path_to_weights = 'saved_models/CnnModel/weights.best.data-200.hdf5'
model.load_weights(path_to_weights)

# Accumulators for count of non-double superframes classified as speaker 1 or 2...
# Accumulate of number of diarization superframe labels that are wrong.
totalNonDoubleTalkSuperframesCount = 0
totalSpeakerLabelErrors = 0

totalRecordingsCount = len(recordingsDirs)
recordingCounter = 0

for recordingDir in recordingsDirs:
    recordingCounter += 1
    print("Processing recording %d of %d" % (recordingCounter, totalRecordingsCount))
    print('Recording dir: %s' % recordingDir)

    # Get the superframe file names and corresponding speaker labels for this recording.
    superframeFiles, trueSpeakerLabels = GetSuperframesInfo.get_super_frame_info(recordingDir)

    # For each superframe file:
    #    read the superframe
    #    run CNN to get the softmax vector (i.e the "activations" or "Speaker embeddings)
    #    add this vector to the features[] array of speaker embeddings
    featuresList = []
    for superframeFile in superframeFiles:
        tensor = LoadDataset.path_to_tensor(recordingDir + '/' + superframeFile).astype('float32')
        # Get embedded speakers feature vector for ths superframe.
        # This is just the last activation layer of the CNN: the softmax predictions.
        emSpeakers = model.predict(tensor)[0]
        featuresList.append(emSpeakers)
        # print("max emSpeaker index %d  " % np.argmax(emSpeakers), "score  %f " % emSpeakers.max())

    # Below, we refer to the embeddedSpeakers vector obtained using the CNN simply as the features.
    # "features" is a 2-D array with the i'th row containing embedded speakers vector for the i'th superframe.
    #  superframe 0  c0, c1, c2, ...., c259
    #  superframe 1  c0, c1, c2, ...., c259
    #  superframe 2  c0, c1, c2, ...., c259
    #      .
    #      .
    #      .
    #  superframe(N-1) c0, c1, c2, ...., c259
    #
    features = np.array(featuresList)
    #print(features.shape)

    if len(features) == 0:
        print("No features generated. Skipping this recording.")
    else:
        features = GetSpeakerLabels.NormalizeFeatureVectors(features)

        speakerLabels = GetSpeakerLabels.CalculateSpeakerLabels(features, newSpeakerThreshold, sameSpeakerThreshold)

        print("Speaker Labels: %d" % len(speakerLabels))
        # print(speakerLabels)
        # print(trueSpeakerLabels)

        """
        Compare the detected speaker labels with the true speaker labels:
            For each true speaker:
                Iterate over superframes labeled as the true speaker.
                Collect a dictionary of (detectedSpeaker, count) pairs for the superframes of that speaker.
                Ignore double-talk superframes that were labelled 9999 for double talk. 
            
            For each true speaker take the detected speaker with the largest count.
            Set this as the count of correct detections for this true speaker.
            
            Repeat for each true speaker.  Because we are only taking superframes labeled with the current true speaker,
            we are not double counting any detections. 
        
            The accuracy is the sum of correct detections / total number of non-double talk superframes.
            Error = 1 - accuracy.
        """

        correctSpeakerDetections = {}

        for speaker in range(1, trueSpeakerCount + 1):
            # Collect results for the current speaker "speaker"
            detections = {}
            # Iterate from 0 to speaker labels count-1 inclusive.
            for i in range(0, len(speakerLabels)):
                trueLabel = trueSpeakerLabels[i]

                # Collect results for this speaker (e.g. 1 or 2 ...) only
                if trueLabel == speaker:
                    # Count only if not doubleTalk.
                    if trueLabel != doubleTalkLabel:
                        label = speakerLabels[i]
                        if label in detections:
                            detections[label] += 1
                        else:
                            detections[label] = 1

            # Get largest count and use this as the detections for this speaker.
            if len(detections) > 0:
                highestCountLabel = max(detections, key=detections.get)
                detCount = detections[highestCountLabel]
                correctSpeakerDetections[speaker] = detCount

        nonDoubleTalkLabelsCount = sum(label != doubleTalkLabel for label in trueSpeakerLabels)
        correctLabelsCount = sum(correctSpeakerDetections.values())

        if nonDoubleTalkLabelsCount > 0:
            acc = float(correctLabelsCount) / float(nonDoubleTalkLabelsCount)

            # print("Accuracy = %f %%" % 100.0 * acc)
            error = (1.0 - acc) * 100.0
            print("Error    = %f %%" % error)

            totalNonDoubleTalkSuperframesCount += nonDoubleTalkLabelsCount
            totalSpeakerLabelErrors += nonDoubleTalkLabelsCount - correctLabelsCount
        else:
            print("Count of non-double talk superframes is 0.  Skipping.")

        print("----------------------------------")

overAllError = totalSpeakerLabelErrors / float(totalNonDoubleTalkSuperframesCount) * 100.0
print("Total number recordings processed       : %d" % len(recordingsDirs))
print("Total number of superframes diarized    : %d" % totalNonDoubleTalkSuperframesCount)
print("Total number of superframes label errors: %d" % totalSpeakerLabelErrors)
print("Total overall diarization error         : %f  %%" % overAllError)
