import LoadDataset
import CalculateTestAccuracy

import R_CnnModel_1

"""
Load test data.
"""
test_files_path = 'C:/diarization/Data-200/test'   # 'data/test'

number_of_targets = 260   # total number of speakers

print("Loading: test_files, test_targets")
test_files, test_targets, speaker_target_names2 = LoadDataset.load_dataset(test_files_path, number_of_targets)
test_tensors = LoadDataset.paths_to_tensor(test_files).astype('float32')



"""
Load the checkpoint model that had the best validation loss.
Measure the accuracy of the classification using the test data set. 
"""
path_to_weights = 'saved_models/r-cnn-1.weights.best.data-200.hdf5'

model = R_CnnModel_1.create_model(number_of_targets, test_tensors.shape, 0.2)
model.summary()
model.load_weights(path_to_weights)

"""
Calculate the accuracy for the test data and print to console.
"""
CalculateTestAccuracy.CalculateTestAccuracy(model, test_targets, test_tensors)

