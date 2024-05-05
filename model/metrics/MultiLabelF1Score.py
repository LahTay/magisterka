import tensorflow as tf


class MultiLabelF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_outputs, name='multi_output_f1score', **kwargs):
        super(MultiLabelF1Score, self).__init__(name=name, **kwargs)
        self.num_outputs = num_outputs
        self.f1_scores = self.add_weight(name='f1_scores', shape=(num_outputs,), initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.precision_objects = [tf.keras.metrics.Precision() for _ in range(num_outputs)]
        self.recall_objects = [tf.keras.metrics.Recall() for _ in range(num_outputs)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(y_pred.shape[-1]):
            output_true = tf.cast(y_true[:, i], tf.float32)
            output_pred = y_pred[:, i]
            self.precision_objects[i].update_state(output_true, output_pred)
            self.recall_objects[i].update_state(output_true, output_pred)

    def result(self):
        all_f1_scores = []
        for i in range(self.num_outputs):
            precision = self.precision_objects[i].result()
            recall = self.recall_objects[i].result()
            f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
            all_f1_scores.append(f1_score)
        average_f1_score = tf.reduce_mean(all_f1_scores)
        metrics_dict = {f'f1_score_{i}': all_f1_scores[i] for i in range(self.num_outputs)}
        metrics_dict['Average_f1_score'] = average_f1_score

        return metrics_dict

    def reset_state(self):
        for p, r in zip(self.precision_objects, self.recall_objects):
            p.reset_state()
            r.reset_state()

    def get_config(self):
        base_config = super(MultiLabelF1Score, self).get_config()
        return {**base_config, "num_outputs": self.num_outputs}
