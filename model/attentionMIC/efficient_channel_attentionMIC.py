from tensorflow.keras.layers import (Layer, Input, GlobalAveragePooling2D, Reshape, Conv1D, Conv2D, Dense,
                                     BatchNormalization, multiply)
from tensorflow import sigmoid
from tensorflow.keras.models import Model
from math import log2

from model.attentionMIC.attentionMIC import AttentionLayer


class ECALayer(Layer):
    """
    ECA paper proposes adaptive kernel size based on the following equation:
    k = floor(log2(C)) + b,
    C - number of channels
    b - hyperparameter (usually 1)
    """

    def __init__(self, gamma=2, b=1, **kwargs):
        super(ECALayer, self).__init__(**kwargs)
        self.gamma = gamma
        self.b = b

    def build(self, input_shape):
        self.kernel_size = int(abs((log2(input_shape[-1]) + self.b) / self.gamma))
        self.kernel_size = max(self.kernel_size, 3)  # Kernel size must be at least 3
        self.conv1d = Conv1D(filters=input_shape[-1],
                             kernel_size=self.kernel_size,
                             padding='same',
                             use_bias=False)
        self.gap = GlobalAveragePooling2D()
        self.reshape1 = Reshape((1, -1))
        super(ECALayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.gap(inputs)
        x = self.reshape1(x)

        x = self.conv1d(x)

        x = Reshape((-1, 1, inputs.shape[-1]))(x)
        x = sigmoid(x)

        output = multiply([inputs, x])
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ECALayer, self).get_config()
        config.update({"gamma": self.gamma})
        config.update({"b": self.b})
        return config


def create_attentionMIC_ECA_model(input_shape, num_classes, filters):
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

    x = AttentionLayer(filters=64)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

