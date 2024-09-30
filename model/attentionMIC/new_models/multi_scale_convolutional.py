from tensorflow.keras import layers, models


def build_multi_scale_cnn_model(input_shape, num_classes, filters):
    inputs = layers.Input(shape=input_shape)

    def build_conv_layers(x):
        # Parallel Convolutional Layers
        conv1 = layers.Conv2D(filters, (1, 1), padding='same')(x)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.ReLU()(conv1)

        conv3 = layers.Conv2D(filters, (3, 3), padding='same')(x)
        conv3 = layers.BatchNormalization()(conv3)
        conv3 = layers.ReLU()(conv3)

        conv5 = layers.Conv2D(filters, (5, 5), padding='same')(x)
        conv5 = layers.BatchNormalization()(conv5)
        conv5 = layers.ReLU()(conv5)

        # Concatenate Feature Maps
        x = layers.concatenate([conv1, conv3, conv5], axis=-1)
        return x

    x = build_conv_layers(inputs)
    x = build_conv_layers(x)

    # Output Layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
