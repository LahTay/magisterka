from tensorflow.keras.layers import (Layer, Input, GlobalAveragePooling2D, Conv2D, Dense, BatchNormalization, multiply)
from tensorflow.keras.models import Model
from model.attentionMIC.efficient_channel_attentionMIC import ECALayer
from model.attentionMIC.time_frequency_attentionMIC import TimeFreqAttentionLayer

def create_attentionMIC_ECA_TFQ_hybrid_model(input_shape, num_classes, filters):
    input_layer = Input(shape=input_shape)

    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = ECALayer()(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = ECALayer()(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = ECALayer()(x)

    x = TimeFreqAttentionLayer(filters=64)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model
