from tensorflow.keras.layers import (Layer, Input, Conv2D, BatchNormalization, Dense, GlobalAveragePooling2D, multiply,
                                     concatenate, Reshape)
from tensorflow.keras.models import Model
import tensorflow as tf


class AttentionLayer(Layer):
    def __init__(self, filters, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.conv1 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation='sigmoid')

    def call(self, inputs, **kwargs):
        feature_maps = self.conv1(inputs)
        attention_maps = self.conv2(inputs)
        attended = multiply([feature_maps, attention_maps])
        return attended

    def get_config(self):
        return super(AttentionLayer, self).get_config()



class MultiHeadAttentionLayer(Layer):
    def __init__(self, num_heads=3, head_size=16, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.attention_heads = [Conv2D(filters=head_size, kernel_size=(1, 1), padding='same', activation='sigmoid') for
                                _ in range(num_heads)]

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        feature_maps = []
        for head in self.attention_heads:
            attention_map = head(inputs)
            feature_map = multiply([inputs, attention_map])
            feature_maps.append(feature_map)

        feature_maps = concatenate(feature_maps, axis=-1)
        feature_maps = Reshape((batch_size, -1, self.num_heads * self.head_size))(feature_maps)
        return feature_maps

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({"num_heads": self.num_heads, "head_size": self.head_size})
        return config

def create_attentionMIC_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)

    # x = MultiHeadAttentionLayer(head_size=32, num_heads=4)(x)
    x = AttentionLayer(filters=64)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model
