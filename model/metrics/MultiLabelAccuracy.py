import tensorflow as tf

class MultiLabelAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_outputs, output_names=None, threshold=0.5, name='multi_output_accuracy', **kwargs):
        super(MultiLabelAccuracy, self).__init__(name=name, **kwargs)
        if output_names is None:
            self.output_names = list(range(num_outputs))
        else:
            self.output_names = output_names
        self.num_outputs = num_outputs
        self.accuracies = self.add_weight(name='accuracies', shape=(num_outputs,), initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        individual_accuracies = []
        for i in range(y_pred.shape[-1]):
            output_true = tf.cast(y_true[:, i], tf.float32)
            output_pred = y_pred[:, i]
            accuracy = tf.keras.metrics.binary_accuracy(output_true, output_pred, self.threshold)
            individual_accuracies.append(accuracy)

        self.accuracies.assign_add(individual_accuracies)
        self.count.assign_add(1)

    def result(self):
        all_accuracies = self.accuracies / self.count
        average_accuracy = tf.reduce_mean(all_accuracies)
        metrics_dict = {f'accuracy_{self.output_names[i]}': all_accuracies[i] for i in range(self.num_outputs)}
        metrics_dict['Average_accuracy'] = average_accuracy

        return metrics_dict

    def reset_state(self):
        self.accuracies.assign(tf.zeros(self.num_outputs))
        self.count.assign(0)

    def get_config(self):
        base_config = super(MultiLabelAccuracy, self).get_config()
        return {**base_config, "num_outputs": self.num_outputs, "threshold": self.threshold}
