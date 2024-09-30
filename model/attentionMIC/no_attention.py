from tensorflow.keras.layers import (Layer, Input, Conv2D, BatchNormalization, Dense, GlobalAveragePooling2D, multiply,
                                     concatenate, Reshape)
from tensorflow.keras.models import Model
import tensorflow as tf

def create_no_attentionMIC_model(input_shape, num_classes, filters):
    input_layer = Input(shape=input_shape)

    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model
