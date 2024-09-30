import tensorflow as tf
from tensorflow.keras import layers, models


class SEBlock(layers.Layer):
    def __init__(self, channels, reduction=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(channels // reduction, activation='relu')
        self.dense2 = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        se = self.global_avg_pool(inputs)
        se = self.dense1(se)
        se = self.dense2(se)
        se = tf.reshape(se, [-1, 1, 1, self.channels])
        x = layers.multiply([inputs, se])
        return x

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({
            'channels': self.channels,
            'reduction': self.reduction
        })
        return config


def build_se_model(input_shape, num_classes, filters):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(3):
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = SEBlock(channels=filters)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
