import tensorflow as tf


class MultiLabelSpecificity(tf.keras.metrics.Metric):
    def __init__(self, num_outputs, output_names=None, threshold=0.5, name='multi_output_specificity', **kwargs):
        super(MultiLabelSpecificity, self).__init__(name=name, **kwargs)
        if output_names is None:
            self.output_names = list(range(num_outputs))
        else:
            self.output_names = output_names
        self.num_outputs = num_outputs
        self.threshold = threshold
        self.specificity_objects = [(tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives()) for _ in range(num_outputs)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_outputs):
            output_true = tf.cast(y_true[:, i], tf.float32)
            output_pred = tf.cast(tf.greater_equal(y_pred[:, i], self.threshold), tf.float32)
            tn_metric, fp_metric = self.specificity_objects[i]
            tn_metric.update_state(output_true, output_pred)
            fp_metric.update_state(output_true, output_pred)

    def result(self):
        individual_specificities = []
        for i in range(self.num_outputs):
            tn_metric, fp_metric = self.specificity_objects[i]
            tn = tn_metric.result()
            fp = fp_metric.result()
            specificity = tn / (tn + fp + tf.keras.backend.epsilon())
            individual_specificities.append(specificity)
        average_specificity = tf.reduce_mean(individual_specificities)
        metrics_dict = {f'specificity_{self.output_names[i]}': individual_specificities[i] for i in range(self.num_outputs)}
        metrics_dict['Average_specificity'] = average_specificity
        return metrics_dict

    def reset_state(self):
        for tn_metric, fp_metric in self.specificity_objects:
            tn_metric.reset_states()
            fp_metric.reset_states()

    def get_config(self):
        base_config = super(MultiLabelSpecificity, self).get_config()
        return {**base_config, "num_outputs": self.num_outputs, "threshold": self.threshold}
