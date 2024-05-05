from tensorflow.keras.layers import (Layer, Input, GlobalAveragePooling2D, Conv2D, Dense, BatchNormalization, multiply)
from tensorflow.keras.models import Model


class TimeFreqAttentionLayer(Layer):
    def __init__(self, filters, **kwargs):
        super(TimeFreqAttentionLayer, self).__init__(**kwargs)
        self.conv1 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation='sigmoid')
        self.conv3 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation='sigmoid')

    def call(self, inputs, **kwargs):
        feature_maps = self.conv1(inputs)
        attention_time = self.conv2(inputs)
        attention_freq = self.conv3(inputs)
        attended = multiply([feature_maps, attention_time, attention_freq])
        return attended

    def get_config(self):
        return super(TimeFreqAttentionLayer, self).get_config()


def create_attentionMIC_TFQ_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)

    x = TimeFreqAttentionLayer(filters=64)(x)

    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model
