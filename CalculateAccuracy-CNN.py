import numpy as np
import CnnModel
import CnnModel_2
import LoadDataset
import CalculateTestAccuracy

"""
Load the checkpoint model that had the best validation loss.
Measure the accuracy of the classification using the test data set. 
"""

path_to_weights = 'saved_models/CnnModel/weights.best.data-200.hdf5'

number_of_targets = 260   # total number of speakers
model = CnnModel_2.create_model(number_of_targets)
model.summary()
model.load_weights(path_to_weights)

test_files_path = 'C:/diarization/Data-200/test'   # 'data/test'

print("Loading: test_files, test_targets")
test_files, test_targets, speaker_target_names2 = LoadDataset.load_dataset(test_files_path, number_of_targets)
test_tensors = LoadDataset.paths_to_tensor(test_files).astype('float32')

"""
Calculate the accuracy for the test data and print to console.
"""

CalculateTestAccuracy.CalculateTestAccuracy(model, test_targets, test_tensors)

