from tensorflow.keras import layers, models


def build_dilated_conv_model(input_shape, num_classes, filters):
    inputs = layers.Input(shape=input_shape)
    dilation_rates = [1, 2, 4]
    x = inputs
    for rate in dilation_rates:
        x = layers.Conv2D(filters, (3, 3), dilation_rate=rate)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
