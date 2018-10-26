from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from tqdm import tqdm  # just a progress bar
import SampleScalingParameters as Scaling

# define function to load train, test, and validation datasets
def load_dataset(path, number_of_targets):
    _data = load_files(path)
    _speaker_files = np.array(_data['filenames'])
    # speaker_targets = np.array(data['target'])
    _speaker_targets = np_utils.to_categorical(np.array(_data['target']), number_of_targets)
    _speaker_target_names = np.array(_data["target_names"])
    return _speaker_files, _speaker_targets, _speaker_target_names


"""
Convert to tensors
Based on similar code in "Udacity ML dog project"
Applies scaling to superframe.
"""


def path_to_tensor(sample_path):
    # Load this speech sample which is stored row-wise as 64 rows with 32 mel cepstrum coefficients per row

    with open(sample_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)

    scaled_data = [(val + Scaling.shift) * Scaling.scale for val in data]

    array = np.reshape(scaled_data, [64, 32])

    # Convert 2D tensor to 4D tensor with shape (1, 64, 32, 1) and return 4D tensor
    a2 = np.expand_dims(array, axis=0)
    a3 = np.expand_dims(a2, axis=3)
    return a3


def paths_to_tensor(sample_paths):
    list_of_tensors = [path_to_tensor(sample_path) for sample_path in tqdm(sample_paths)]
    return np.vstack(list_of_tensors)

