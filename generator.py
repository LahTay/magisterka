import math
import numpy as np
from tensorflow.keras.utils import Sequence
import audiofile
import librosa
from tqdm import tqdm
"""
How the Generator class is supposed to work
It's going to be used in the .fit function of keras.

__init__(x_paths, y_values, batch_size):
    x_paths - full paths to each audio file
    y_values - list of lists of labels where each index of the 1st list corresponds to the same index of x_paths
    Simplest model - those lists are multi-label binary encoded (example. [0, 1, 1, 0, 0, 1]
    Harder model - not implemented
    batch_size = 64
    indices - which audio data to read

__len__():
    math.ceil(len(self.x) / self.batch_size)


__get_item__(idx):
    This should 1st read audio files.
    You get batch_size amount of filenames from x_paths. Those filenames are decided by indices array.
    Then you read those filenames,
    assign them to an array, 
    add corresponding labels of those audio data,
    return them.


on_epoch_end():
    randomize indices so that each epoch has different batches

"""

class AudioGenerator(Sequence):
    """
    A Keras Sequence generator for audio data. Supports various audio feature extractions
    like MFCC, spectrogram, chromagram, and multichannel spectrogram.

    Parameters:
    - x_paths: ndarray of audio file paths.
    - y_values: ndarray of labels corresponding to x_paths.
    - batch_size: Integer, size of each data batch.
    - feature_type: String, type of feature to compute ('raw', 'mfcc', 'spectrogram',
                    'chromagram', 'multichannel_spectrogram').
    - n_fft: Integer, FFT window size.
    - hop_length: Integer, hop length for STFT.
    - n_mfcc: Integer, number of MFCC coefficients to compute.
    - n_chroma: Integer, number of chroma bins.
    - multi_label: Boolean, whether the labels are multi-label.
    - classes_names: List of class names for multi-label outputs.
    - testing: Boolean, whether the generator is used for testing.
    - preprocess: Boolean, whether to preprocess data upfront.
    - normalize: Boolean, whether to normalize features.
    - train_mean: Precomputed mean for normalization.
    - train_std: Precomputed standard deviation for normalization.
    - audio_bits: Integer, bit depth of audio files.
    - resample_sr: Integer, target sampling rate for resampling.
    """
    def __init__(self, x_paths, y_values, batch_size=64, feature_type="mfcc", n_fft=1024, hop_length=512, n_mfcc=40,
                 multi_label=True, classes_names=None, testing=False, preprocess=True, normalize=True, train_mean=None,
                 train_std=None, audio_bits=None, resample_sr=None, n_chroma=12, pool_spectrogram=False,
                 pool_type='average', pool_size=(2, 2)):
        self.x = x_paths.flatten()  # Has to be ndarray
        self.y = y_values  # Has to be ndarray
        self.testing = testing
        self.batch_size = batch_size
        self.feature_type = feature_type.lower()
        valid_features = ['raw', 'mfcc', 'spectrogram', 'chromagram', 'multichannel_spectrogram']
        temp_disabled = ['multichannel_spectrogram']
        if self.feature_type not in valid_features or self.feature_type in temp_disabled:
            raise ValueError(f"Invalid feature_type: {self.feature_type}. "
                             f"Valid options are {valid_features}.")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        # Temporary setting to have different generator output for multilabel model
        self.multi_label = multi_label
        self.classes_names = classes_names
        self.testing = testing
        self.preprocess = preprocess
        self.normalize = normalize
        self.train_mean = train_mean
        self.train_std = train_std
        self.audio_bits = audio_bits
        self.resample_sr = resample_sr

        # Pooling parameters
        self.pool_spectrogram = pool_spectrogram
        self.pool_type = pool_type.lower()
        self.pool_size = pool_size

        self.indices = np.arange(len(self.x))
        self.len = math.ceil(self.x.shape[0] / self.batch_size)
        self.data_shape = self._get_data_shape()

        if self.preprocess:
            self.preprocessed_data = np.zeros((len(self.x), *self.data_shape), dtype=np.float32)
            self._preprocess_data()
        else:
            self.preprocessed_data = None

        self._shuffle_indices()
        print(f"Generator initialized with {len(self.x)} audio files.")
        print(f"Total batches per epoch: {self.len}")
        print(f"Audio data has a shape of {self.data_shape}")

    def _get_data_shape(self):
        sample_data, sr = self._read_audio(self.x[0], self.resample_sr)
        feature = self._compute_feature(sample_data, sr, max_frames=None)
        return feature.shape

    def _preprocess_data(self):
        for i, file_name in tqdm(enumerate(self.x), total=len(self.x),
                                 desc=f"Preprocessing {'Testing' if self.testing else 'Training'} Audio"):
            audio_data, sr = self._read_audio(file_name, self.resample_sr)
            processed_data = self._compute_feature(audio_data, sr, max_frames=self.data_shape[1])
            self.preprocessed_data[i] = processed_data

        if self.normalize:
            if self.train_mean is None or self.train_std is None:
                print("Calculating new means and stds (it can take long time if using spectrogram)")
                self.train_mean = np.mean(self.preprocessed_data, axis=(0, 2), keepdims=True)
                self.train_std = np.std(self.preprocessed_data, axis=(0, 2), keepdims=True)

            self.preprocessed_data = (self.preprocessed_data - self.train_mean) / (self.train_std + 1e-8)

            # self.train_mean = np.squeeze(np.mean(self.preprocessed_data, axis=(0, 2), keepdims=True))
            # self.train_std = np.squeeze(np.std(self.preprocessed_data, axis=(0, 2), keepdims=True))

            # self.preprocessed_data = ((self.preprocessed_data - self.train_mean[None, :, None]) /
            #                           (self.train_std[None, :, None] + 1e-8))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ids = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[ids]

        if self.preprocess:
            batch_x = self.preprocessed_data[ids]

        else:
            batch_x_names = self.x[ids]
            batch_x = np.zeros((len(ids), *self.data_shape), dtype=np.float32)

            for i, name in enumerate(batch_x_names):
                data, sr = self._read_audio(name)
                batch_x[i] = self._compute_feature(data, sr, max_frames=self.data_shape[1])

        if self.multi_label:
            batch_y = {f'{self.classes_names[i].replace(" ", "_")}': batch_y[:, i:i + 1]
                            for i in range(len(self.classes_names))}

        return batch_x, batch_y

    def on_epoch_end(self):
        self._shuffle_indices()

    def _shuffle_indices(self):
        if not self.testing:
            np.random.shuffle(self.indices)

    def _read_audio(self, file_name, target_sr=None):
        dtype_map = {8: 'int8', 16: 'int16', 32: 'int32'}
        dtype = dtype_map.get(self.audio_bits, None)
        audio_data, sr = audiofile.read(file_name, dtype=dtype)
        if target_sr is not None and sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return audio_data, sr

    def _compute_feature(self, data, sr, max_frames=None):
        if len(data) < self.n_fft:
            data = np.pad(data, (0, self.n_fft - len(data)), mode='constant')

        if self.feature_type == 'mfcc':
            feature = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft,
                                           hop_length=self.hop_length, win_length=self.n_fft, window='hamming')
        elif self.feature_type == 'spectrogram':
            S = librosa.stft(data, n_fft=self.n_fft, hop_length=self.hop_length)
            feature = np.abs(S)
            feature = self._apply_pooling(feature)
        elif self.feature_type == 'chromagram':
            feature = librosa.feature.chroma_stft(y=data, sr=sr, n_fft=self.n_fft,
                                                  hop_length=self.hop_length, n_chroma=self.n_chroma)
        elif self.feature_type == 'multichannel_spectrogram':
            feature = self._compute_multichannel_spectrogram(data, sr)
            feature = self._apply_pooling(feature)
        else:  # 'raw'
            feature = data

        if max_frames is not None:
            feature = self._pad_or_truncate(feature, max_frames)
        return feature

    def _pad_or_truncate(self, feature, max_frames):
        if feature.ndim == 1:
            if len(feature) < max_frames:
                pad_width = max_frames - len(feature)
                feature = np.pad(feature, (0, pad_width), mode='constant')
            else:
                feature = feature[:max_frames]
        else:
            if feature.shape[1] < max_frames:
                pad_width = max_frames - feature.shape[1]
                feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
            else:
                feature = feature[:, :max_frames]
        return feature

    def _apply_pooling(self, feature):
        if self.pool_spectrogram:
            pool_size_freq, pool_size_time = self.pool_size

            if feature.ndim == 2:
                # For 2D features (e.g., spectrogram)
                feature = self._pooling_2d(feature, pool_size_freq, pool_size_time, self.pool_type)
            elif feature.ndim == 3:
                # For 3D features (e.g., multichannel spectrogram)
                feature = self._pooling_3d(feature, pool_size_freq, pool_size_time, self.pool_type)
        return feature

    def _pooling_2d(self, data, pool_size_freq, pool_size_time, pool_type):
        freq_bins, time_frames = data.shape
        new_freq = freq_bins // pool_size_freq
        new_time = time_frames // pool_size_time
        data = data[:new_freq * pool_size_freq, :new_time * pool_size_time]
        data = data.reshape(new_freq, pool_size_freq, new_time, pool_size_time)
        if pool_type == 'max':
            data = data.max(axis=(1, 3))
        else:  # 'average'
            data = data.mean(axis=(1, 3))
        return data

    def _pooling_3d(self, data, pool_size_freq, pool_size_time, pool_type):
        freq_bins, time_frames, channels = data.shape
        new_freq = freq_bins // pool_size_freq
        new_time = time_frames // pool_size_time
        data = data[:new_freq * pool_size_freq, :new_time * pool_size_time, :]
        data = data.reshape(new_freq, pool_size_freq, new_time, pool_size_time, channels)
        if pool_type == 'max':
            data = data.max(axis=(1, 3))
        else:  # 'average'
            data = data.mean(axis=(1, 3))
        return data

    def _compute_multichannel_spectrogram(self, audio_data, sr):
        S = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        S_delta = librosa.feature.delta(S_db)
        S_delta2 = librosa.feature.delta(S_db, order=2)

        harmonic, percussive = librosa.effects.hpss(audio_data)

        harmonic_stft = librosa.stft(harmonic, n_fft=self.n_fft, hop_length=self.hop_length)
        harmonic_db = librosa.amplitude_to_db(np.abs(harmonic_stft), ref=np.max)

        percussive_stft = librosa.stft(percussive, n_fft=self.n_fft, hop_length=self.hop_length)
        percussive_db = librosa.amplitude_to_db(np.abs(percussive_stft), ref=np.max)

        # Ensure all features have the same shape
        min_frames = min(S_db.shape[1], S_delta.shape[1], S_delta2.shape[1],
                         harmonic_db.shape[1], percussive_db.shape[1])
        S_db = S_db[:, :min_frames]
        S_delta = S_delta[:, :min_frames]
        S_delta2 = S_delta2[:, :min_frames]
        harmonic_db = harmonic_db[:, :min_frames]
        percussive_db = percussive_db[:, :min_frames]

        multi_channel_spectrogram = np.stack([S_db, S_delta, S_delta2, harmonic_db, percussive_db], axis=-1)

        return multi_channel_spectrogram

    def get_label_num(self):
        """
        Assumes the y values are in the same shape as if transformed by the MultiLabelBinarizer
        from sklearn.preprocessing

        Returns: How many labels there are in a generator

        """
        return self.y.shape[1]
