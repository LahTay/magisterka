import tensorflow as tf
from tensorflow.keras import layers, models


def build_combined_conv_model(input_shape, num_classes, filters1d, filters2d):
    inputs = layers.Input(shape=input_shape)

    # 2D Convolutional Layers
    x = layers.Conv2D(filters2d, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters2d, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Reshape to Apply 1D Convolutions
    x_shape = x.shape
    x = layers.Reshape((x_shape[2], x_shape[1] * x_shape[3]))(x)

    # 1D Convolutional Layers
    x = layers.Conv1D(filters1d, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters1d, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Output Layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
