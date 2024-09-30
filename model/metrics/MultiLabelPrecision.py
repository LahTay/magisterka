import tensorflow as tf


class MultiLabelPrecision(tf.keras.metrics.Metric):
    def __init__(self, num_outputs, output_names=None, threshold=0.5, name='multi_output_precision', **kwargs):
        super(MultiLabelPrecision, self).__init__(name=name, **kwargs)
        if output_names is None:
            self.output_names = list(range(num_outputs))
        else:
            self.output_names = output_names
        self.num_outputs = num_outputs
        self.threshold = threshold
        self.precision_objects = [tf.keras.metrics.Precision(thresholds=threshold) for _ in range(num_outputs)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_outputs):
            output_true = tf.cast(y_true[:, i], tf.float32)
            output_pred = tf.cast(y_pred[:, i], tf.float32)
            self.precision_objects[i].update_state(output_true, output_pred)

    def result(self):
        individual_precisions = [p.result() for p in self.precision_objects]
        average_precision = tf.reduce_mean(individual_precisions)
        metrics_dict = {f'precision_{self.output_names[i]}': individual_precisions[i] for i in range(self.num_outputs)}
        metrics_dict['Average_precision'] = average_precision
        return metrics_dict

    def reset_state(self):
        for p in self.precision_objects:
            p.reset_state()

    def get_config(self):
        base_config = super(MultiLabelPrecision, self).get_config()
        return {**base_config, "num_outputs": self.num_outputs,
                "threshold": self.threshold}
