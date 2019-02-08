"""
Given an input binary file (or array) containing an array of cepstrum frames:
- Organise the frames into superframes (with overlap)
- Run the superframes array in sequence through the CNN/RNN Diarizer with clustering
- Given the superframe labelling by speaker, generate a labelling per input frame.
- Return the frame labelling as an array.
"""
import numpy as np
import SampleScalingParameters as Scaling
import CnnModel_2
import Clustering

"""
Read in the cepstrum frames and convert them to overlapping superframes.
"""


def read_frames_into_superframes_with_scaling(input_cepstrums_file_path, frame_len, superframe_len, frame_overlap):
    # Load frames of this speech sample which is stored row-wise as superframe_len rows with frame_len mel cepstrum coefficients per row

    with open(input_cepstrums_file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)

    # Apply scaling as was applied to training data when the CNN-RNN models were created.
    scaled_data = [(val + Scaling.shift) * Scaling.scale for val in data]

    frameCount = int(len(scaled_data) / frame_len)
    superframeCount = int(frameCount / frame_overlap) - 1



    # The frames should be read back using overlapping superframes.
    # Using speech features only, create superframes.
    #
    # | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --  cepstrum frames
    # \----------- /
    #      \----------- /
    #            \----------- /
    # Superframes overlap  by 25% for 2.048s super frame.
    #                      By 50% for 0.512s superframe.
    #                      By 50% for 1.024s superframe

    # Build superframes by taking K = # of frames per superframe, and advancing the row position by frame_overlap
    frames = np.reshape(scaled_data, [frameCount, frame_len])

    super_frames_list = []
    frame_counter = 0
    for rowPos in range(0, (frameCount - superframe_len), frame_overlap):
        super_frame = np.zeros((superframe_len, frame_len), dtype=np.float32)
        for i in range(0, superframe_len):
            super_frame[i] = frames[rowPos + i]
            frame_counter += 1
        super_frames_list.append(super_frame)

    print(frame_counter)
    s = super_frames_list[0]
    print(s.shape)
    print(len(super_frames_list))
    print("frameCount = %d  actual superframeCount = %d" % (frameCount, len(super_frames_list)))

    return super_frames_list


###########################################################################################################################
"""
Given an input binary file (or array) containing an array of cepstrum frames:
- Organise the frames into superframes (with overlap)
- Run the superframes array in sequence through the CNN/RNN Diarizer with clustering
- Given the superframe labelling by speaker, generate a labelling per input frame.
- Return the frame labelling as an array.
"""
# Set a fixed random generator seed to get repeatable results.
r = 1
from numpy.random import seed
seed(r)
from tensorflow import set_random_seed
set_random_seed(r)

# Setup input.
cepstrumFramesFilePath = 'E:/RepoExperiments/JsiDiarize/Output/SpeechFeatures.bin'
frame_len = 32
super_frame_len = 16  # 16 for 0.5s, 32 for 1.0s superframe
frame_overlap = (int)(0.5 * super_frame_len)
maxSpeakersClasses = 3

output_superframe_labels_file_path = 'E:/RepoExperiments/JsiDiarize/Output/superframe_labels.bin'
output_frame_labels_file_path = 'E:/RepoExperiments/JsiDiarize/Output/frame_labels.bin'

#
# Read input frames and organize input superframes. --------------------------------------------------------
#
superframes_list = read_frames_into_superframes_with_scaling(cepstrumFramesFilePath, frame_len, super_frame_len, frame_overlap)
print("superframes_list count = %d" % len(superframes_list))

#
# Convert superframes to tensors ----------------------------------------------------------------------------
#
tensors_list = []
for sp in superframes_list:
    # Convert 2D superframe to 4D tensor with shape (1, super_frame_len, frame_len, 1) and return 4D tensor
    a2 = np.expand_dims(sp, axis=0)
    a3 = np.expand_dims(a2, axis=3)
    tensors_list.append(a3)

tensors = np.array(tensors_list)
print(tensors.shape)

#
# Setup the CNN model ------------------------------------------------------------------------------------------
#
dropout_rate = 0  # not used here
tensor_shape = (1, super_frame_len, frame_len, 1)
number_of_targets = 260  # total number of speakers
"""
# Best CNN only model -- Use output of Flatten layer instead of 260 speaker scores.
fullmodel = CnnModel_2.create_model(number_of_targets, tensor_shape, dropout_rate, n_frames)
path_to_weights = 'saved_models_halfSec/CnnModel/cnn-weights.Best.data-1000.hdf5'
fullmodel.load_weights(path_to_weights)
layer_name = 'flatten_layer'
model = Model(inputs=fullmodel.input, outputs=fullmodel.get_layer(layer_name).output)
model.summary()
"""

# Best CNN only model -- Output 260 speaker scores.
model = CnnModel_2.create_model(number_of_targets, tensor_shape, dropout_rate, super_frame_len)
path_to_weights = 'saved_models_halfSec/CnnModel/cnn-weights.Best.data-1000.hdf5'
model.load_weights(path_to_weights)
model.summary()

"""
# Best CNN only model -- Output 260 speaker scores.
model = CnnModel_2.create_model(number_of_targets, tensor_shape, dropout_rate, super_frame_len)
path_to_weights = 'saved_models_1sec/CnnModel/cnn-weights.TEST.data-250.hdf5'
model.load_weights(path_to_weights)
model.summary()
"""

#
# Run the CNN to get the features vectors (one per superframe tensor) -----------------------------------------------
#
# For each superframe:
#    read the superframe
#    run CNN to get the softmax vector (i.e the "activations" or "Speaker embeddings)
#    add this vector to the features[] array of speaker embeddings
featuresList = []
for tensor in tensors:
    # Get embedded speakers feature vector for this superframe tensor
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
print(features.shape)
# print(features)

speakerLabels = []
if len(features) < 4:
    print("Less than minimum number of features generated. Skipping this recording.")
else:
    # Use Gaussian Mixture clustering.  Determine the number of classes by increasing the number of GMM
    # components starting at 2 and stopping when the silouette distortion increases.
    clusteredFeatures, distortion, clustersCount = Clustering.runGaussianMixtureClustering(features, maxSpeakersClasses)

    print("Detected Clusters: %d  Distortion: %f" % (clustersCount, distortion))

    clusteredFeatures += 1  # make the labels start at 1.
    speakerLabels = clusteredFeatures.tolist()

    print("Speaker Labels: %d" % len(speakerLabels))
    print(speakerLabels)


#
#  Convert superframe labels to frame labels
#
#
# | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | --  cepstrum frames
# \---------/
#      \---------/
#            \---------/
# Superframes overlap  50 % for 1.024s and 0.512s superframe options.
#
superframe_count = len(speakerLabels)
total_frame_count = superframe_count * frame_overlap + frame_overlap
# Assign the superframe label to the frames in the first half of the superframe.
frame_labels = []

for label in speakerLabels:
    for i in range(0, frame_overlap):
        frame_labels.append(label)

for i in range(0, frame_overlap):
    frame_labels.append(speakerLabels[superframe_count - 1])

#
# Save the frame speaker labels to a binary file of 32 bit integers.
#
framelabels_array = np.array(frame_labels)
fh = open(output_frame_labels_file_path, "bw")
framelabels_array.tofile(fh)
print("frameLabels written to %s" % (output_frame_labels_file_path))


#
# Save the superframe speaker labels to a binary file of 32 bit integers.
#
speakerlabels_array = np.array(speakerLabels)
fh = open(output_superframe_labels_file_path, "bw")
speakerlabels_array.tofile(fh)
print("speakerLabels written to %s" % (output_superframe_labels_file_path))

print(framelabels_array.shape)
