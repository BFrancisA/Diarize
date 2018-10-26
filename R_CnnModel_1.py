from keras.layers import Bidirectional, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, Input, GRU, Dense, Activation, Dropout, Reshape, Flatten, CuDNNGRU
from keras.models import Model


"""
Create model: CNN + RNN 
"""


def create_model(speaker_target_names_count, input_tensors_shape, dropout_rate):
    """
    Create model
    Reference: https://github.com/sharathadavanne/multichannel-sed-crnn/blob/master/sed.py
    The above reference is to create a similar CNN RNN model used for bird audio detection.
    This reference is used as a starting point for this model design.
    """

    data_in_shape = input_tensors_shape  # (1, 64, 32, 1)

    # RNN-CNN specification
    #                            3rd from last   2nd from last         last
    spec_input = Input(shape=(data_in_shape[-3], data_in_shape[-2], data_in_shape[-1]))
    spec = spec_input

    spec = Conv2D(filters=32, kernel_size=3, padding='same')(spec)
    spec = Activation('relu')(spec)
    # spec = MaxPooling2D(pool_size=2, strides=(2, 2))(spec)
    spec = Dropout(dropout_rate)(spec)

    spec = Conv2D(filters=32, kernel_size=3, padding='same')(spec)
    spec = Activation('relu')(spec)
    spec = MaxPooling2D(pool_size=2, strides=(2, 2))(spec)
    spec = Dropout(dropout_rate)(spec)

    spec = Conv2D(filters=64, kernel_size=3, padding='same')(spec)
    spec = Activation('relu')(spec)
    spec = MaxPooling2D(pool_size=2, strides=(2, 2))(spec)
    spec = Dropout(dropout_rate)(spec)

    spec = Conv2D(filters=64, kernel_size=3, padding='same')(spec)
    spec = Activation('relu')(spec)
    spec = MaxPooling2D(pool_size=2, strides=(2, 2))(spec)
    spec = Dropout(dropout_rate)(spec)

    # Convert 8 x 4 x 64 to 32 x 64 by concatenating rows
    spec = Reshape((-1, 64))(spec)

    # Is multiplying the output of forward and reverse directions teh right thing to do here?
    # Sum appears to be best
    spec = Bidirectional(GRU(32, activation='tanh', dropout=dropout_rate,
                             recurrent_dropout=dropout_rate, return_sequences=True),
                         merge_mode='sum')(spec)

    spec = Bidirectional(GRU(32, activation='tanh', dropout=dropout_rate,
                             recurrent_dropout=dropout_rate, return_sequences=True),
                         merge_mode='sum')(spec)

    # spec = Bidirectional(CuDNNGRU(32, return_sequences=True),
    #                      merge_mode='sum')(spec)
    # spec = Dropout(dropout_rate)(spec)
    #
    # spec = Bidirectional(CuDNNGRU(32, return_sequences=True),
    #                      merge_mode='sum')(spec)
    # spec = Dropout(dropout_rate)(spec)

    # Are these needed?
    spec = TimeDistributed(Dense(32))(spec)
    spec = Dropout(dropout_rate)(spec)

    spec = TimeDistributed(Dense(32))(spec)
    spec = Dropout(dropout_rate)(spec)

    spec = Flatten()(spec)
    # spec = Dense(128, activation='relu')(spec)
    spec = Dropout(0.3)(spec)
    out = Dense(speaker_target_names_count, activation='softmax', name='output')(spec)

    model = Model(inputs=spec_input, outputs=out)

    return model

