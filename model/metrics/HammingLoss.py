import tensorflow as tf


class HammingLoss(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='hamming_loss', **kwargs):
        super(HammingLoss, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.hamming_loss = self.add_weight(name='hl', initializer='zeros')
        self.total_samples = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probabilities to binary predictions
        y_pred = tf.cast(y_pred > self.threshold, tf.int32)

        misclassified = tf.cast(y_true != y_pred, tf.float32)

        self.hamming_loss.assign_add(tf.reduce_sum(misclassified))
        self.total_samples.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        # Return the average Hamming loss
        return self.hamming_loss / self.total_samples

    def reset_state(self):
        # Reset the state of the metric
        self.hamming_loss.assign(0)
        self.total_samples.assign(0)