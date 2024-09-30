import tensorflow as tf


class MultiLabelInformedness(tf.keras.metrics.Metric):
    def __init__(self, num_outputs, output_names=None, threshold=0.5, name='multi_output_informedness', **kwargs):
        super(MultiLabelInformedness, self).__init__(name=name, **kwargs)
        if output_names is None:
            self.output_names = list(range(num_outputs))
        else:
            self.output_names = output_names
        self.num_outputs = num_outputs
        self.threshold = threshold
        self.recall_objects = [tf.keras.metrics.Recall(thresholds=threshold) for _ in range(num_outputs)]
        self.specificity_objects = [(tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives()) for _ in range(num_outputs)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_outputs):
            output_true = tf.cast(y_true[:, i], tf.float32)
            output_pred = tf.cast(tf.greater_equal(y_pred[:, i], self.threshold), tf.float32)
            self.recall_objects[i].update_state(output_true, output_pred)
            tn_metric, fp_metric = self.specificity_objects[i]
            tn_metric.update_state(output_true, output_pred)
            fp_metric.update_state(output_true, output_pred)

    def result(self):
        individual_informedness_scores = []
        for i in range(self.num_outputs):
            recall = self.recall_objects[i].result()
            tn_metric, fp_metric = self.specificity_objects[i]
            tn = tn_metric.result()
            fp = fp_metric.result()
            specificity = tn / (tn + fp + tf.keras.backend.epsilon())
            informedness = recall + specificity - 1
            individual_informedness_scores.append(informedness)
        average_informedness = tf.reduce_mean(individual_informedness_scores)
        metrics_dict = {f'informedness_{self.output_names[i]}': individual_informedness_scores[i] for i in range(self.num_outputs)}
        metrics_dict['Average_informedness'] = average_informedness
        return metrics_dict

    def reset_state(self):
        for recall, (tn_metric, fp_metric) in zip(self.recall_objects, self.specificity_objects):
            recall.reset_states()
            tn_metric.reset_states()
            fp_metric.reset_states()

    def get_config(self):
        base_config = super(MultiLabelInformedness, self).get_config()
        return {**base_config, "num_outputs": self.num_outputs, "threshold": self.threshold}
