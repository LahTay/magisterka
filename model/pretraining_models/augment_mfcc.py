import tensorflow as tf
from tensorflow.keras.layers import Layer

class AugmentMFCCs(Layer):
    def __init__(self, time_shift_ratio=0.1, freq_mask_ratio=0.1, freq_mask_num = 1,min_warp_ratio=0.05,
                 max_warp_ratio=0.25, max_no_warp=2, **kwargs):
        super(AugmentMFCCs, self).__init__(**kwargs)
        self.time_shift_ratio = time_shift_ratio
        self.freq_mask_ratio = freq_mask_ratio
        self.freq_mask_num = freq_mask_num
        self.min_warp_ratio = min_warp_ratio
        self.max_warp_ratio = max_warp_ratio
        self.max_number_warp = max_no_warp
        if self.min_warp_ratio > self.max_warp_ratio:
            raise ValueError("min_warp_ratio cannot be greater than time_warp_ratio.")

    def call(self, inputs, **kwargs):
        # Time shift
        if 0 < self.time_shift_ratio < 1:
            shift = tf.cast(inputs.shape[2] * self.time_shift_ratio, tf.int32)
            inputs = tf.roll(inputs, shift=shift, axis=2)

        # Frequency masking
        if 0 < self.freq_mask_ratio < 1:
            for _ in range(self.freq_mask_num):
                freq_mask_param = tf.random.uniform([], minval=0, maxval=int(inputs.shape[2] * self.freq_mask_ratio),
                                                    dtype=tf.int32)
                inputs = self._frequency_mask(inputs, freq_mask_param)

        # Time warping
        if 0 < self.max_warp_ratio < 1:
            inputs = self._time_warp(inputs)

        return inputs

    def _frequency_mask(self, inputs, mask_size):
        # Frequency masking by setting certain frequency bands to zero
        f0 = tf.random.uniform([], minval=0, maxval=inputs.shape[2] - mask_size, dtype=tf.int32)
        mask = tf.concat([
            tf.ones((inputs.shape[0], inputs.shape[1], f0)),
            tf.zeros((inputs.shape[0], inputs.shape[1], mask_size)),
            tf.ones((inputs.shape[0], inputs.shape[1], inputs.shape[2] - f0 - mask_size))
        ], axis=-1)
        return inputs * mask

    def _time_warp(self, inputs):
        batch_size, freq_bins, time_steps = inputs.shape

        num_warp_points = tf.random.uniform([], minval=1, maxval=self.max_number_warp+1, dtype=tf.int32)

        # Initialize src_indices as the identity (no warping)
        src_indices = tf.range(time_steps)
        dst_indices = tf.identity(src_indices)

        for _ in range(num_warp_points):
            # Randomly choose a center point to apply the warping around
            center = tf.random.uniform([], minval=int(time_steps * 0.1), maxval=int(time_steps * 0.5), dtype=tf.int32)

            # Determine the warp range as a random percentage between a range of time_steps
            warp_range_percentage = tf.random.uniform([], minval=self.min_warp_ratio, maxval=self.max_warp_ratio,
                                                      dtype=tf.float32)
            warp_range = tf.cast(tf.round(time_steps * warp_range_percentage), tf.int32)

            # Ensure that the warp doesn't go out of bounds
            start_idx = tf.maximum(0, center - warp_range // 2)
            end_idx = tf.minimum(time_steps, start_idx + warp_range)

            max_distance = time_steps - end_idx - warp_range - 1
            warp_distance = tf.random.uniform([], minval=1, maxval=max_distance + 1, dtype=tf.int32)

            # Swapping the selected segments
            dst_indices = tf.concat([
                dst_indices[:start_idx],  # Keep the beginning of the sequence unchanged
                dst_indices[start_idx + warp_distance:start_idx + warp_distance + warp_range],  # Part to move earlier
                dst_indices[start_idx + warp_range:start_idx + warp_distance],
                # Sequence in between that stays unchanged
                dst_indices[start_idx:start_idx + warp_range],  # Part to move later
                dst_indices[start_idx + warp_range + warp_distance:]  # The rest of the sequence unchanged
            ], axis=0)

        # Apply the warping along the time axis
        warped_inputs = tf.gather(inputs, dst_indices, axis=2)
        return warped_inputs

    def get_config(self):
        config = super(AugmentMFCCs, self).get_config()
        config.update({
            "time_shift_ratio": self.time_shift_ratio,
            "freq_mask_ratio": self.freq_mask_ratio,
            "time_warp_ratio": self.time_warp_ratio,
        })
        return config
