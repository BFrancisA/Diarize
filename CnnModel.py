from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

"""
Create model
"""


def create_model(speaker_target_names_count, input_tensors_shape, dropout_rate):
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(64, 32, 1)))
    model.add(MaxPooling2D(pool_size=2, strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(speaker_target_names_count, activation='softmax'))

    return model

