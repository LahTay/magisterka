import tensorflow as tf


class MultiLabelMarkedness(tf.keras.metrics.Metric):
    def __init__(self, num_outputs, output_names=None, threshold=0.5, name='multi_output_markedness', **kwargs):
        super(MultiLabelMarkedness, self).__init__(name=name, **kwargs)
        if output_names is None:
            self.output_names = list(range(num_outputs))
        else:
            self.output_names = output_names
        self.num_outputs = num_outputs
        self.threshold = threshold
        self.precision_objects = [tf.keras.metrics.Precision(thresholds=threshold) for _ in range(num_outputs)]
        self.inverse_precision_objects = [(tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalseNegatives()) for _ in range(num_outputs)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_outputs):
            output_true = tf.cast(y_true[:, i], tf.float32)
            output_pred = tf.cast(tf.greater_equal(y_pred[:, i], self.threshold), tf.float32)
            self.precision_objects[i].update_state(output_true, output_pred)
            tn_metric, fn_metric = self.inverse_precision_objects[i]
            tn_metric.update_state(output_true, output_pred)
            fn_metric.update_state(output_true, output_pred)

    def result(self):
        individual_markedness_scores = []
        for i in range(self.num_outputs):
            precision = self.precision_objects[i].result()
            tn_metric, fn_metric = self.inverse_precision_objects[i]
            tn = tn_metric.result()
            fn = fn_metric.result()
            inverse_precision = tn / (tn + fn + tf.keras.backend.epsilon())
            markedness = precision + inverse_precision - 1
            individual_markedness_scores.append(markedness)
        average_markedness = tf.reduce_mean(individual_markedness_scores)
        metrics_dict = {f'markedness_{self.output_names[i]}': individual_markedness_scores[i] for i in range(self.num_outputs)}
        metrics_dict['Average_markedness'] = average_markedness
        return metrics_dict

    def reset_state(self):
        for precision, (tn_metric, fn_metric) in zip(self.precision_objects, self.inverse_precision_objects):
            precision.reset_states()
            tn_metric.reset_states()
            fn_metric.reset_states()

    def get_config(self):
        base_config = super(MultiLabelMarkedness, self).get_config()
        return {**base_config, "num_outputs": self.num_outputs, "threshold": self.threshold}
