from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf


class MultiOutputAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_outputs, name='multi_output_accuracy', **kwargs):
        super(MultiOutputAccuracy, self).__init__(name=name, **kwargs)
        self.num_outputs = num_outputs
        self.accuracies = self.add_weight(name='accuracies', shape=(num_outputs,), initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        individual_accuracies = []
        for i in range(y_pred.shape[-1]):
            output_true = tf.cast(y_true[:, i], tf.float32)
            output_pred = y_pred[:, i]
            accuracy = tf.keras.metrics.binary_accuracy(output_true, output_pred)
            individual_accuracies.append(accuracy)

        self.accuracies.assign_add(individual_accuracies)
        self.count.assign_add(1)

    def result(self):
        all_accuracies = self.accuracies / self.count
        average_accuracy = tf.reduce_mean(all_accuracies)
        metrics_dict = {f'accuracy_{i}': all_accuracies[i] for i in range(self.num_outputs)}
        metrics_dict['Average_accuracy'] = average_accuracy

        return metrics_dict

    def reset_state(self):
        # Reset the state of the metric
        self.accuracies.assign(tf.zeros(self.num_outputs))
        self.count.assign(0)

    def get_config(self):
        base_config = super(MultiOutputAccuracy, self).get_config()
        return {**base_config, "num_outputs": self.num_outputs}

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
    x = Dense(units=1, activation='sigmoid', name=f"{name}_output_dense")(x)
    return x


def create_conv_model_from_paper(input_shape, num_classes, outputs_names):
    # This is the shared input layer
    assert len(outputs_names) == num_classes
    input_layer = Input(shape=input_shape)

    # Create a unified submodel for each class and store the outputs
    outputs = [create_unified_submodel(input_layer, name=name) for name in outputs_names]

    # If you have multiple classes, concatenate the outputs. Otherwise, just use a single output.
    if num_classes > 1:
        final_output = concatenate(outputs, axis=-1, name="concatenate_output")
    else:
        final_output = outputs[0]

    # Create the final model
    final_model = Model(inputs=input_layer, outputs=final_output)

    return final_model
