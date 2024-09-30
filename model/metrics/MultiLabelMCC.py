import tensorflow as tf


class MultiLabelMCC(tf.keras.metrics.Metric):
    def __init__(self, num_outputs, output_names=None, threshold=0.5, name='multi_output_mcc', **kwargs):
        super(MultiLabelMCC, self).__init__(name=name, **kwargs)
        if output_names is None:
            self.output_names = list(range(num_outputs))
        else:
            self.output_names = output_names
        self.num_outputs = num_outputs
        self.threshold = threshold
        self.mcc_scores = [(tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
                            tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives())
                           for _ in range(num_outputs)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_outputs):
            output_true = tf.cast(y_true[:, i], tf.float32)
            output_pred = tf.cast(tf.greater_equal(y_pred[:, i], self.threshold), tf.float32)
            tp_metric, tn_metric, fp_metric, fn_metric = self.mcc_scores[i]
            tp_metric.update_state(output_true, output_pred)
            tn_metric.update_state(output_true, output_pred)
            fp_metric.update_state(output_true, output_pred)
            fn_metric.update_state(output_true, output_pred)

    def result(self):
        individual_mcc_scores = []
        for i in range(self.num_outputs):
            tp_metric, tn_metric, fp_metric, fn_metric = self.mcc_scores[i]
            tp = tp_metric.result()
            tn = tn_metric.result()
            fp = fp_metric.result()
            fn = fn_metric.result()
            numerator = tp * tn - fp * fn
            denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = numerator / (denominator + tf.keras.backend.epsilon())
            individual_mcc_scores.append(mcc)
        average_mcc = tf.reduce_mean(individual_mcc_scores)
        metrics_dict = {f'mcc_{self.output_names[i]}': individual_mcc_scores[i] for i in range(self.num_outputs)}
        metrics_dict['Average_mcc'] = average_mcc
        return metrics_dict

    def reset_state(self):
        for tp_metric, tn_metric, fp_metric, fn_metric in self.mcc_scores:
            tp_metric.reset_states()
            tn_metric.reset_states()
            fp_metric.reset_states()
            fn_metric.reset_states()

    def get_config(self):
        base_config = super(MultiLabelMCC, self).get_config()
        return {**base_config, "num_outputs": self.num_outputs, "threshold": self.threshold}
