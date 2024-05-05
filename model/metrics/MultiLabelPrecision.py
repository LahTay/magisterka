import tensorflow as tf


class MultiLabelPrecision(tf.keras.metrics.Metric):
    def __init__(self, num_outputs, name='multi_output_precision', **kwargs):
        super(MultiLabelPrecision, self).__init__(name=name, **kwargs)
        self.num_outputs = num_outputs
        self.precision_objects = [tf.keras.metrics.Precision() for _ in range(num_outputs)]
        self.precisions = self.add_weight(name='precisions', shape=(num_outputs,), initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(y_pred.shape[-1]):
            output_true = tf.cast(y_true[:, i], tf.float32)
            output_pred = tf.round(y_pred[:, i])
            self.precision_objects[i].update_state(output_true, output_pred)

    def result(self):
        all_precisions = [p.result() for p in self.precision_objects]
        average_precision = tf.reduce_mean(all_precisions)
        metrics_dict = {f'precision_{i}': all_precisions[i] for i in range(self.num_outputs)}
        metrics_dict['Average_precision'] = average_precision

        return metrics_dict

    def reset_state(self):
        for p in self.precision_objects:
            p.reset_state()

    def get_config(self):
        base_config = super(MultiLabelPrecision, self).get_config()
        return {**base_config, "num_outputs": self.num_outputs}
