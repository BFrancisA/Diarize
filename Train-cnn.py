"""
Train 2D-CNN Speaker Classifier
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import optimizers
import LoadDataset
import CnnModel
import CnnModel_2


"""
Load the training data for the speaker classifier
"""

# train_files_path = 'Data-Small/train'
# valid_files_path = 'Data-Small/valid'
# test_files_path  = 'Data-Small/test'
# number_of_targets = 5  # total number of speakers

# train_files_path = 'C:/Diarization/0.5-sec-superframes/Data-Small/train'
# valid_files_path = 'C:/Diarization/0.5-sec-superframes/Data-Small/valid'
# test_files_path  = 'C:/Diarization/0.5-sec-superframes/Data-Small/test'
# number_of_targets = 5  # total number of speakers

# train_files_path = 'C:/Diarization/0.5-sec-superframes/Data-1000/train'
# valid_files_path = 'C:/Diarization/0.5-sec-superframes/Data-1000/valid'
# test_files_path  = 'C:/Diarization/0.5-sec-superframes/Data-1000/test'

train_files_path = 'C:/Diarization/1-sec-superframes/Data-250/train'
valid_files_path = 'C:/Diarization/1-sec-superframes/Data-250/valid'
test_files_path  = 'C:/Diarization/1-sec-superframes/Data-250/test'

number_of_targets = 260   # total number of speakers


print("Loading: train_files, train_targets")
train_files, train_targets, speaker_target_names = LoadDataset.load_dataset(train_files_path, number_of_targets)

print("Loading: valid_files, valid_targets")
valid_files, valid_targets, speaker_target_names3 = LoadDataset.load_dataset(valid_files_path, number_of_targets)

print("Loading: test_files, test_targets")
test_files, test_targets, speaker_target_names2 = LoadDataset.load_dataset(test_files_path, number_of_targets)

train_files_count = len(train_files)
test_files_count = len(test_files)
valid_files_count = len(valid_files)

speaker_target_names_count = len(speaker_target_names)

print( "Total number of training files   : %d" % train_files_count)
print( "Total number of test files       : %d" % test_files_count)
print( "Total number of validation files : %d" % valid_files_count)

print( "Total number of target names     : %d" % speaker_target_names_count)
print("")

for i in range(0, (len(speaker_target_names))):
    print("speaker target name  : %s" % speaker_target_names[i])

print("")

for i in range(0, 3):  # range(len(speaker_targets)):
    print("train_files[%d]       : %s" % (i, train_files[i]))
    print("train_targets[%d]     : %s" % (i, train_targets[i]))
    print("")
    

"""
Convert the data from arrays of superframes to 4-D tensors,
"""
train_tensors = LoadDataset.paths_to_tensor(train_files).astype('float32')
valid_tensors = LoadDataset.paths_to_tensor(valid_files).astype('float32')
test_tensors  = LoadDataset.paths_to_tensor(test_files).astype('float32')

print("Train:")
print(len(train_tensors))
print(train_tensors.shape)

print("Test:")
print(len(test_tensors) )
print(test_tensors.shape )

print("Valid")
print(len(valid_tensors) )
print(valid_tensors.shape )


"""
Create model
"""
dropout_rate = 0.2
data_in_shape = train_tensors.shape  # 2.048s (1, 64, 32, 1)  or  0.512s (1, 16, 32, 1)

n_frames = 32  # frames per superframe   16 for 0.5sec superframe, 32 for 1.0sec superframe

#model = CnnModel.create_model(number_of_targets, train_tensors.shape, dropout_rate)
model = CnnModel_2.create_model(number_of_targets, train_tensors.shape, dropout_rate, n_frames)

model.summary()

"""
Compile model
"""
# default is lr=0.001, use lr=0.0001 for large training set
rmsprop_slow = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer=rmsprop_slow, loss='categorical_crossentropy', metrics=['accuracy'])

#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""
Train model
"""
epochs = 1000  # 1000

checkpointer = ModelCheckpoint(filepath='saved_models_1sec/CnnModel/cnn-weights.TEST.data-250.hdf5',
                               verbose=1, save_best_only=True)


earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='auto')

history = model.fit(train_tensors, train_targets,
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=200, verbose=0,
          callbacks=[checkpointer, earlystopping])


"""
Save the model and the last generated weights separately.
"""
# # Architecture
# with open("saved_models_halfSec/cnnModel_2TEST.json", "w") as f:
#     f.write(model.to_json())
# f.close()
#
# # Weights
# model.save_weights("saved_models_halfSec/model.TEST.hdf5")

"""
Load the checkpoint model that had the best validation loss.
Measure the accuracy of the classified using the test data set. 
"""
model.load_weights('saved_models_1sec/CnnModel/cnn-weights.TEST.data-250.hdf5')

speaker_classifications = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# Calculate accuracy
test_accuracy = 100*np.sum(np.array(speaker_classifications) == np.argmax(test_targets, axis=1))/len(speaker_classifications)
print('Test accuracy: %.4f%%' % test_accuracy)
print(test_accuracy)

"""
Generate the model accuracy and model loss plots from the model fit history.
Ref. https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
"""
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('saved_models_1sec/CnnModel/cnn-weights.TEST.data-250.history.accuracy.svg')
plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('saved_models_1sec/CnnModel/cnn-weights.TEST.data-250.history.loss.svg')
plt.show()
