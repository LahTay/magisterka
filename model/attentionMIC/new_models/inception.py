from tensorflow.keras import layers, models


def build_inception_model(input_shape, num_classes, filters1, filters2):
    inputs = layers.Input(shape=input_shape)

    def inception_block(x):
        # Inception Module
        branch1 = layers.Conv2D(filters1, (1, 1), padding='same', activation='relu')(x)

        branch2 = layers.Conv2D(filters1, (1, 1), padding='same', activation='relu')(x)
        branch2 = layers.Conv2D(filters2, (3, 3), padding='same', activation='relu')(branch2)

        branch3 = layers.Conv2D(filters1, (1, 1), padding='same', activation='relu')(x)
        branch3 = layers.Conv2D(filters2, (5, 5), padding='same', activation='relu')(branch3)

        branch4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch4 = layers.Conv2D(filters2, (1, 1), padding='same', activation='relu')(branch4)

        # Concatenate Outputs
        x = layers.concatenate([branch1, branch2, branch3, branch4], axis=-1)
        return x

    x = inception_block(inputs)
    x = inception_block(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
