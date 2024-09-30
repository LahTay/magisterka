from tensorflow.keras import layers, models

def build_tcn_model(input_shape, num_classes, filters):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    dilation_rates = [1, 2, 4, 8]
    for rate in dilation_rates:
        x = layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=(1, rate))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
