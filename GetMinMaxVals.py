
from sklearn.datasets import load_files
import numpy as np


"""
Load the training data for the speaker classifier
"""
train_files_path = 'C:/diarization/Data/train'  # 'data/train'
valid_files_path = 'C:/diarization/Data/valid'  # 'data/valid'
test_files_path  = 'C:/diarization/Data/test'   # 'data/test'


def load_dataset(path):
    _data = load_files(path)
    _speaker_files = np.array(_data['filenames'])
    return _speaker_files


def get_min_max(files_list, min, max ):
    for filePath in files_list:
        with open(filePath, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
        for val in data:
            if val < min:
                min = val
            if val > max:
                max = val
    return min, max


minval = np.finfo(np.float32).max
maxval = -np.finfo(np.float32).max


print("Loading: test_files")
test_files = load_dataset(test_files_path)
minval, maxval = get_min_max(test_files, minval, maxval)
print("Min feature val: ", minval)
print("Max feature val: ", maxval)
print()

print("Loading: valid_files")
valid_files = load_dataset(valid_files_path)
minval, maxval = get_min_max(valid_files, minval, maxval)
print("Min feature val: ", minval)
print("Max feature val: ", maxval)
print()

print("Loading: train_files")
train_files  = load_dataset(train_files_path)
minval, maxval = get_min_max(train_files, minval, maxval)
print("Min feature val: ", minval)
print("Max feature val: ", maxval)
print()

print("Final Min feature val: ", minval)
print("Final Max feature val: ", maxval)

