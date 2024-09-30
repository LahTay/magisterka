import tensorflow as tf

class MultiLabelCohenKappa(tf.keras.metrics.Metric):
    def __init__(self, num_outputs, output_names=None, threshold=0.5, name='multi_output_cohen_kappa', **kwargs):
        super(MultiLabelCohenKappa, self).__init__(name=name, **kwargs)
        if output_names is None:
            self.output_names = list(range(num_outputs))
        else:
            self.output_names = output_names
        self.num_outputs = num_outputs
        self.threshold = threshold
        self.cohen_kappa_scores = [(tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()) for _ in range(num_outputs)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_outputs):
            output_true = tf.cast(y_true[:, i], tf.float32)
            output_pred = tf.cast(tf.greater_equal(y_pred[:, i], self.threshold), tf.float32)
            tp_metric, tn_metric, fp_metric, fn_metric = self.cohen_kappa_scores[i]
            tp_metric.update_state(output_true, output_pred)
            tn_metric.update_state(output_true, output_pred)
            fp_metric.update_state(output_true, output_pred)
            fn_metric.update_state(output_true, output_pred)

    def result(self):
        individual_cohen_kappa_scores = []
        for i in range(self.num_outputs):
            tp_metric, tn_metric, fp_metric, fn_metric = self.cohen_kappa_scores[i]
            tp = tp_metric.result()
            tn = tn_metric.result()
            fp = fp_metric.result()
            fn = fn_metric.result()
            numerator = 2 * (tp * tn - fp * fn)
            denominator = (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)
            cohen_kappa = numerator / (denominator + tf.keras.backend.epsilon())
            individual_cohen_kappa_scores.append(cohen_kappa)
        average_cohen_kappa = tf.reduce_mean(individual_cohen_kappa_scores)
        metrics_dict = {f'cohen_kappa_{self.output_names[i]}': individual_cohen_kappa_scores[i] for i in range(self.num_outputs)}
        metrics_dict['Average_cohen_kappa'] = average_cohen_kappa
        return metrics_dict

    def reset_state(self):
        for tp_metric, tn_metric, fp_metric, fn_metric in self.cohen_kappa_scores:
            tp_metric.reset_states()
            tn_metric.reset_states()
            fp_metric.reset_states()
            fn_metric.reset_states()

    def get_config(self):
        base_config = super(MultiLabelCohenKappa, self).get_config()
        return {**base_config, "num_outputs": self.num_outputs, "threshold": self.threshold}
