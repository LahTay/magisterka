import tensorflow as tf


class MultiLabelF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_outputs, output_names=None, threshold=0.5, name='multi_output_f1_score', **kwargs):
        super(MultiLabelF1Score, self).__init__(name=name, **kwargs)
        if output_names is None:
            self.output_names = list(range(num_outputs))
        else:
            self.output_names = output_names
        self.num_outputs = num_outputs
        self.threshold = threshold
        self.precision_objects = [tf.keras.metrics.Precision(thresholds=threshold) for _ in range(num_outputs)]
        self.recall_objects = [tf.keras.metrics.Recall(thresholds=threshold) for _ in range(num_outputs)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_outputs):
            output_true = y_true[:, i]
            output_pred = y_pred[:, i]
            self.precision_objects[i].update_state(output_true, output_pred)
            self.recall_objects[i].update_state(output_true, output_pred)

    def result(self):
        individual_f1_scores = []
        for i in range(self.num_outputs):

            precision = self.precision_objects[i].result()
            recall = self.recall_objects[i].result()
            f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
            individual_f1_scores.append(f1)
        average_f1_score = tf.reduce_mean(individual_f1_scores)
        metrics_dict = {f'f1_score_{self.output_names[i]}': individual_f1_scores[i] for i in range(self.num_outputs)}
        metrics_dict['Average_f1_score'] = average_f1_score
        return metrics_dict

    def reset_state(self):
        for precision, recall in zip(self.precision_objects, self.recall_objects):
            precision.reset_states()
            recall.reset_states()

    def get_config(self):
        base_config = super(MultiLabelF1Score, self).get_config()
        return {**base_config, "num_outputs": self.num_outputs, "threshold": self.threshold}