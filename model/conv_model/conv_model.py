from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten, concatenate
from tensorflow.keras.models import Model


def create_conv_block(input_tensor, filters, name, id):
    # Apply 2D convolution, max pooling and batch normalization
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', name=f"{name}_conv_{id}")(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name=f"{name}_max_pool_{id}")(x)
    x = BatchNormalization(name=f"{name}_bn_{id}")(x)
    return x


def create_dense_block(input_tensor, units_list, name):
    # Apply a series of dense layers as specified in units_list
    for id, units in enumerate(units_list):
        input_tensor = Dense(units=units, activation='relu', name=f"{name}_dense_{id}")(input_tensor)
    return input_tensor


def create_unified_submodel(input_tensor, name):
    # Convolutional blocks
    name = name.replace(" ", "_")
    x = create_conv_block(input_tensor, filters=128, name=name, id=0)  # Block 1
    x = create_conv_block(x, filters=62, name=name, id=1)  # Block 2
    x = create_conv_block(x, filters=32, name=name, id=2)  # Block 3
    x = Flatten(name=f"{name}_flatten")(x)
    x = create_dense_block(x, units_list=[64, 32, 16], name=name)
    x = Dense(units=1, activation='sigmoid', name=f"{name}")(x)
    return x


def create_conv_model_from_paper(input_shape, num_classes, outputs_names):
    # This is the shared input layer

    assert len(outputs_names) == num_classes, (f"Assertion failed: len(outputs_names) "
                                               f"({len(outputs_names)}) != num_classes ({num_classes})")
    input_layer = Input(shape=input_shape)

    outputs = [create_unified_submodel(input_layer, name=name) for name in outputs_names]

    final_model = Model(inputs=input_layer, outputs=outputs)

    return final_model
