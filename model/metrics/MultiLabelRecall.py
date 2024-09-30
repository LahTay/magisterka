import tensorflow as tf


class MultiLabelRecall(tf.keras.metrics.Metric):
    def __init__(self, num_outputs, output_names=None, threshold=0.5, name='multi_output_recall', **kwargs):
        super(MultiLabelRecall, self).__init__(name=name, **kwargs)
        if output_names is None:
            self.output_names = list(range(num_outputs))
        else:
            self.output_names = output_names
        self.num_outputs = num_outputs
        self.threshold = threshold
        self.recall_objects = [tf.keras.metrics.Recall(thresholds=threshold) for _ in range(num_outputs)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_outputs):
            output_true = y_true[:, i]
            output_pred = y_pred[:, i]
            self.recall_objects[i].update_state(output_true, output_pred)

    def result(self):
        individual_recalls = [rec.result() for rec in self.recall_objects]
        average_recall = tf.reduce_mean(individual_recalls)
        metrics_dict = {f'recall_{self.output_names[i]}': individual_recalls[i] for i in range(self.num_outputs)}
        metrics_dict['Average_recall'] = average_recall
        return metrics_dict

    def reset_state(self):
        for recall in self.recall_objects:
            recall.reset_states()

    def get_config(self):
        base_config = super(MultiLabelRecall, self).get_config()
        return {**base_config, "num_outputs": self.num_outputs, "threshold": self.threshold}
