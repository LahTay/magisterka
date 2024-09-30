from tensorflow.keras.layers import (Layer, Input, GlobalAveragePooling2D, Conv2D, Dense, BatchNormalization, multiply,
                                     Activation, concatenate)
from tensorflow.keras.models import Model

import tensorflow_model_optimization as tfmot
ConstantSparsity = tfmot.sparsity.keras.ConstantSparsity
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
import tensorflow as tf


LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer
class TimeFreqAttentionQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights
    def get_weights_and_quantizers(self, layer):
        return [
            (layer.conv1.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=True)),
            (layer.conv2.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=True)),
            (layer.conv3.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=True))
        ]

    # Configure how to quantize activations
    def get_activations_and_quantizers(self, layer):
        return [
            (layer.conv1.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False)),
            (layer.conv2.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False)),
            (layer.conv3.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))
        ]

    # Setting quantized weights
    def set_quantize_weights(self, layer, quantize_weights):
        layer.conv1.kernel = quantize_weights[0]
        layer.conv2.kernel = quantize_weights[1]
        layer.conv3.kernel = quantize_weights[2]

    # Setting quantized activations
    def set_quantize_activations(self, layer, quantize_activations):
        layer.conv1.activation = quantize_activations[0]
        layer.conv2.activation = quantize_activations[1]
        layer.conv3.activation = quantize_activations[2]

    # Configure output quantization if necessary (can leave empty if not needed)
    def get_output_quantizers(self, layer):
        return []

    # Config serialization for Keras model saving/loading
    def get_config(self):
        return {}


class TimeFreqAttentionLayer(Layer):
    def __init__(self, filters, **kwargs):
        super(TimeFreqAttentionLayer, self).__init__(**kwargs)
        self.filters = filters
        self.conv1 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=filters, kernel_size=(1, 5), padding='same', activation='sigmoid')
        self.conv3 = Conv2D(filters=filters, kernel_size=(5, 1), padding='same', activation='sigmoid')

    def call(self, inputs, **kwargs):
        feature_maps = self.conv1(inputs)
        attention_time = self.conv2(inputs)
        attention_freq = self.conv3(inputs)
        attended_time = multiply([feature_maps, attention_time])
        attended = multiply([attended_time, attention_freq])
        return attended

    def get_config(self):
        config = super(TimeFreqAttentionLayer, self).get_config()
        config.update({
            'filters': self.filters
        })
        return config


class MultiHeadAttentionTFQLayer(Layer):
    def __init__(self, num_heads=3, head_size=16, **kwargs):
        super(MultiHeadAttentionTFQLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_size = head_size
        self.feature_extractors = [Conv2D(filters=head_size, kernel_size=(1, 1), padding='same', activation='relu') for _ in range(num_heads)]
        self.attention_heads = [Conv2D(filters=head_size, kernel_size=(1, 3), padding='same', activation='sigmoid') for _ in range(num_heads)]
        self.attention_heads2 = [Conv2D(filters=head_size, kernel_size=(3, 1), padding='same', activation='sigmoid') for _ in range(num_heads)]

    def call(self, inputs, **kwargs):
        feature_maps = []
        for feature_extractor, attention_head1, attention_head2 in zip(self.feature_extractors, self.attention_heads,
                                                                       self.attention_heads2):
            feature_map = feature_extractor(inputs)
            attention_map1 = attention_head1(inputs)
            attention_map2 = attention_head2(inputs)
            time_weighted_feature_map = multiply([feature_map, attention_map1])
            weighted_feature_map = multiply([time_weighted_feature_map, attention_map2])
            feature_maps.append(weighted_feature_map)

        feature_maps = concatenate(feature_maps, axis=-1)
        return feature_maps

    def get_config(self):
        config = super(MultiHeadAttentionTFQLayer, self).get_config()
        config.update({"num_heads": self.num_heads, "head_size": self.head_size})
        return config




def create_tfq_prune(input_shape, num_classes, target_sparsity=0.5, begin_step=0, frequency=100, quantize=False, filters=64):
    pruning_params_1_by_3 = {'sparsity_m_by_n': (1, 9)}
    pruning_params_sparsity = {'pruning_schedule': ConstantSparsity(target_sparsity=target_sparsity,
                                                                    begin_step=begin_step, frequency=frequency)}

    input_layer = Input(shape=input_shape)

    x = (prune_low_magnitude(Conv2D(filters=filters, kernel_size=(3, 3)), **pruning_params_sparsity)
         (input_layer))
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = prune_low_magnitude(Conv2D(filters=filters, kernel_size=(3, 3)), **pruning_params_1_by_3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = prune_low_magnitude(Conv2D(filters=filters, kernel_size=(3, 3)), **pruning_params_1_by_3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if quantize:
        quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
        x = quantize_annotate_layer(TimeFreqAttentionLayer(filters=64), TimeFreqAttentionQuantizeConfig())(x)
    else:
        x = TimeFreqAttentionLayer(filters=64)(x)

    x = GlobalAveragePooling2D()(x)
    x = prune_low_magnitude(Dense(128, activation='relu'), **pruning_params_1_by_3)(x)
    x = prune_low_magnitude(Dense(64, activation='relu'), **pruning_params_sparsity)(x)
    output_layer = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def create_tfq_no_prune(input_shape, num_classes, quantize=False, filters=64):
    input_layer = Input(shape=input_shape)

    x = Conv2D(filters=filters, kernel_size=(3, 3))(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if quantize:
        quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
        x = quantize_annotate_layer(TimeFreqAttentionLayer(filters=64), TimeFreqAttentionQuantizeConfig())(x)
    else:
        x = TimeFreqAttentionLayer(filters=64)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def create_attentionMIC_TFQ_model(input_shape, num_classes, prune=False,
                                  target_sparsity=0.5, begin_step=0, frequency=100, quantize=False, filters=64):

    if prune:
        model = create_tfq_prune(input_shape, num_classes, target_sparsity, begin_step, frequency, quantize, filters)
    else:
        model = create_tfq_no_prune(input_shape, num_classes, quantize, filters)



    return model
