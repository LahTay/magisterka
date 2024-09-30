import tensorflow as tf

class MultiLabelInverseRecall(tf.keras.metrics.Metric):
    def __init__(self, num_outputs, output_names=None, threshold=0.5, name='multi_output_inverse_recall', **kwargs):
        super(MultiLabelInverseRecall, self).__init__(name=name, **kwargs)
        if output_names is None:
            self.output_names = list(range(num_outputs))
        else:
            self.output_names = output_names
        self.num_outputs = num_outputs
        self.threshold = threshold
        self.inverse_recall_objects = [(tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalseNegatives()) for _ in range(num_outputs)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_outputs):
            output_true = tf.cast(y_true[:, i], tf.float32)
            output_pred = tf.cast(tf.greater_equal(y_pred[:, i], self.threshold), tf.float32)
            tn_metric, fn_metric = self.inverse_recall_objects[i]
            tn_metric.update_state(output_true, output_pred)
            fn_metric.update_state(output_true, output_pred)

    def result(self):
        individual_inverse_recalls = []
        for i in range(self.num_outputs):
            tn_metric, fn_metric = self.inverse_recall_objects[i]
            tn = tn_metric.result()
            fn = fn_metric.result()
            inverse_recall = tn / (tn + fn + tf.keras.backend.epsilon())
            individual_inverse_recalls.append(inverse_recall)
        average_inverse_recall = tf.reduce_mean(individual_inverse_recalls)
        metrics_dict = {f'inverse_recall_{self.output_names[i]}': individual_inverse_recalls[i] for i in range(self.num_outputs)}
        metrics_dict['Average_inverse_recall'] = average_inverse_recall
        return metrics_dict

    def reset_state(self):
        for tn_metric, fn_metric in self.inverse_recall_objects:
            tn_metric.reset_states()
            fn_metric.reset_states()

    def get_config(self):
        base_config = super(MultiLabelInverseRecall, self).get_config()
        return {**base_config, "num_outputs": self.num_outputs, "threshold": self.threshold}