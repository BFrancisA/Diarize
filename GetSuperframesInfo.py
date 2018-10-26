import os
import numpy as np

"""
Load the training data for training the RNN.  This is a set of recordings, where each recording 
has already been converted to superframe features.  Each directory holds one audio recording converted 
to a time ordered sequence of super frames.  Each frame is named <superframe index>.<pin>
For example:

RNNTrainingData-Trim 
          /sw_45001               <- the recording dir
               /0001.6300         <- 1st superframe binary file.  Here speaker 6300 is speaking.
               /0002.6300         <- 2nd superframe and 6300 is still speaking.

Here 0001.6300 is name of the first superframe in the recording.  The speaker PIN number is 6300.
For the RNN, we convert the speaker PINs to just speaker 1, speaker2,... We just care about speaker changes.
Usually there are only 2 speakers per recording.
We reserve 0 for the start of the sequence
Speaker i is the i'th speaker
Speaker 9999 is reserved for a double-talk superframe where both speakers are talking at one time.

We need an array of superframes and an array of labels, one label for each superframe in the sequence.

"""



""""
 For the recording dir:
  -  get a list of superframe files and the corresponding speaker  labels 1, 2, or 9999 for double talk.
  
"""


def get_super_frame_info(recoringDir):
    # Get all the superframe file names in one recording dir.
    super_frame_files = [name for name in os.listdir(recoringDir)]
    super_frame_speaker_pins = [f.split('.')[1] for f in super_frame_files]
    #print(super_frame_files)
    #print(super_frame_speaker_pins)

    # Get unique pins in superFrameSpeakerPins.
    unique_speaker_pins = np.unique(super_frame_speaker_pins)

    # Make a speaker labels array.
    # Replace each pin with a label: 1, 2, or 9999.
    speaker_labels = super_frame_speaker_pins
    speaker = 1
    for pin in unique_speaker_pins:
        if pin != '9999':
            speaker_labels = [speaker if val == pin else val  for val in speaker_labels ]
            speaker = speaker + 1
        else:
            speaker_labels = [9999 if val == pin else val for val in speaker_labels]

    #print(unique_speaker_pins)
    #print(speaker_labels)

    return super_frame_files, speaker_labels
