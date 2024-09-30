# Mobile-net style
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def build_mobilenet_resnet_style_model(input_shape, num_classes, filters):
    inputs = layers.Input(shape=input_shape)

    # Initial Conv Layer
    x = layers.SeparableConv2D(filters, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    def separable_residual_block(x, filters):
        shortcut = x
        # First Separable Conv Layer
        x = layers.SeparableConv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        # Second Separable Conv Layer
        x = layers.SeparableConv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        # Add the shortcut (skip connection)
        x = layers.add([shortcut, x])
        x = layers.ReLU()(x)
        return x

    # Stacking Residual Blocks
    x = separable_residual_block(x, filters)
    x = separable_residual_block(x, filters)
    x = separable_residual_block(x, filters)

    # Global Average Pooling and Output Layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model


def build_mobilenet_style_model(input_shape, num_classes, filters):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(6):
        x = layers.SeparableConv2D(filters, (3, 3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model
