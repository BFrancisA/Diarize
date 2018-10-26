from sklearn.datasets import load_files
import matplotlib.pyplot as plt
import numpy as np
import SampleScalingParameters as Scaling


test_files_path = 'C:/diarization/Data-small/test'   # 'data/test'


def load_dataset(path):
    _data = load_files(path)
    _speaker_files = np.array(_data['filenames'])
    return _speaker_files




def MakeSpectrogram(filePath) :
    with open(filePath, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    scaled_data = [(val + Scaling.shift)*Scaling.scale for val in data]
    return np.reshape(scaled_data, [64, 32])



test_files = load_dataset(test_files_path)

spec_0 = MakeSpectrogram(test_files[10])
spec_1 = MakeSpectrogram(test_files[11])
spec_2 = MakeSpectrogram(test_files[12])


fig, ax0 = plt.subplots(figsize=(8, 2))
ax0.imshow(spec_0, cmap='viridis', interpolation='nearest', aspect='auto')

fig, ax1 = plt.subplots(figsize=(8, 2))
ax1.imshow(spec_1, cmap='viridis', interpolation='nearest', aspect='auto')

fig, ax2 = plt.subplots(figsize=(8, 2))
ax2.imshow(spec_2, cmap='viridis', interpolation='nearest', aspect='auto')

plt.tight_layout()
plt.show()

