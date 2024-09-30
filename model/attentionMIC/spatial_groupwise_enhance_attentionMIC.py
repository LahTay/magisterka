from tensorflow.keras.layers import (Layer, Input, GlobalAveragePooling2D, Reshape, Conv2D, Dense, Activation,
                                     BatchNormalization, multiply)
from tensorflow import split, concat
from tensorflow.keras.models import Model
from model.attentionMIC.attentionMIC import AttentionLayer
from tensorflow import sigmoid

class SGELayer(Layer):
    """
    From the paper's description, the SGE layer involves the following steps:

    1. Divide the input feature maps into groups along the channel dimension.
    2. For each group, perform Global Average Pooling (GAP) to squeeze the spatial dimensions.
    3. Use these statistics to calculate spatial attention maps for each group.
    4. Apply the attention maps to the original feature maps.
    5. Merge the enhanced feature maps from all groups.
    """
    def __init__(self, num_groups, **kwargs):
        super(SGELayer, self).__init__(**kwargs)
        self.num_groups = num_groups

    def build(self, input_shape):
        self.c = input_shape[-1]
        assert self.c % self.num_groups == 0, "Number of channels must be divisible by number of groups"
        self.group_channels = self.c // self.num_groups
        self.bn = BatchNormalization()
        self.gap = GlobalAveragePooling2D()
        self.reshape = Reshape((1, 1, self.group_channels))
        super(SGELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_groups = split(inputs, self.num_groups, axis=-1)
        output_groups = []

        for group in input_groups:
            group_squeezed = self.gap(group)
            group_squeezed = self.reshape(group_squeezed)

            initial_attention = multiply([group, group_squeezed])
            bn_attention = self.bn(initial_attention)

            final_attention = sigmoid(bn_attention)

            group_enhanced = multiply([final_attention, group])
            output_groups.append(group_enhanced)

        outputs = concat(output_groups, axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(SGELayer, self).get_config()
        config.update({"num_groups": self.num_groups})
        return config


def create_attentionMIC_SGE_model(input_shape, num_classes, num_groups, filters):
    input_layer = Input(shape=input_shape)
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = SGELayer(num_groups)(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = SGELayer(num_groups)(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = SGELayer(num_groups)(x)

    x = AttentionLayer(filters=64)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model
