import tensorflow as tf
from tensorflow.keras import layers, models


def build_crnn_bidirectional_model(input_shape, num_classes, filters, lstm_units):
    inputs = layers.Input(shape=input_shape)

    # Convolutional Layers
    x = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Reshape for RNN
    # Swap the frequency and time dimensions
    x = layers.Permute((2, 1, 3))(x)
    # Combine frequency and channels
    x_shape = x.shape
    x = layers.Reshape((x_shape[1], x_shape[2] * x_shape[3]))(x)

    # Recurrent Layer
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)

    # Output Layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model


def build_crnn_model(input_shape, num_classes, filters, lstm_units):
    inputs = layers.Input(shape=input_shape)

    # Convolutional Layers
    x = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Reshape for RNN
    # Swap the frequency and time dimensions
    x = layers.Permute((2, 1, 3))(x)
    # Combine frequency and channels
    x_shape = x.shape
    x = layers.Reshape((x_shape[1], x_shape[2] * x_shape[3]))(x)

    # Recurrent Layer
    x = layers.LSTM(lstm_units)(x)

    # Output Layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
