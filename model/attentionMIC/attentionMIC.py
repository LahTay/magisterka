from tensorflow.keras.layers import Layer, Input, Conv2D, BatchNormalization, Dense, GlobalAveragePooling2D, multiply
from tensorflow.keras.models import Model


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


def create_attentionMIC_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(x)
    x = BatchNormalization()(x)

    x = AttentionLayer(filters=64)(x)

    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model
