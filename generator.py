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
    def __init__(self, x_paths, y_values, batch_size=64, use_mfcc=False, n_fft=1024, hop_length=512, n_mfcc=40,
                 multi_label=True, classes_names=None, testing=False, use_spectrogram=False, preprocess=True):
        self.x = x_paths.flatten()  # Has to be ndarray
        self.y = y_values  # Has to be ndarray
        self.testing = testing
        self.batch_size = batch_size
        self.use_mfcc = use_mfcc
        self.use_spectrogram = use_spectrogram
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.indices = np.arange(x_paths.shape[0])
        self.len = math.ceil(self.x.shape[0] / self.batch_size)
        self.multi_label = multi_label  # Temporary setting to have different generator output for multilabel model
        self.classes_names = classes_names
        self.data_shape = self._return_data_shape()
        self.preprocessed_data = np.zeros((len(self.x), *self.data_shape))
        self.preprocess = preprocess
        if self.preprocess:
            self._preprocess_data()
        self._shuffle_indices()
        print(f"Generator initialized with {len(self.x)} audio files.")
        print(f"Total batches per epoch: {self.len}")
        print(f"Audio data has a shape of {self.data_shape}")

    def _return_data_shape(self):
        if self.use_mfcc:
            sample_data, sr = librosa.load(self.x[0], sr=None)
            mfccs = librosa.feature.mfcc(y=sample_data, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft,
                                         hop_length=self.hop_length, win_length=self.n_fft, window='hamming')
            return mfccs.shape
        elif self.use_spectrogram:
            sample_data, sr = librosa.load(self.x[0], sr=None)
            spectrogram = librosa.stft(sample_data, n_fft=self.n_fft, hop_length=self.hop_length)
            return spectrogram.shape
        else:
            sample_data, _ = audiofile.read(self.x[0])
            return sample_data.shape

    def _preprocess_data(self):
        # Preprocess all data and store in self.preprocessed_data
        for i, file_name in tqdm(enumerate(self.x), total=len(self.x),
                                 desc=f"Preprocessing {'Testing' if self.testing else 'Training'} Audio"):
            audio_data, sr = self._read_audio(file_name)
            if self.use_mfcc:
                processed_data = self._compute_mfcc(audio_data, sr, self.data_shape[1])
            elif self.use_spectrogram:
                processed_data = self._compute_spectrogram(audio_data, sr)
            else:
                processed_data = audio_data
            self.preprocessed_data[i] = processed_data

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ids = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[ids]
        if not self.preprocess:
            actual_batch_size = len(ids)
            batch_x_names = self.x[ids]

            if self.use_mfcc or self.use_spectrogram:
                batch_x = np.zeros((actual_batch_size, *self.data_shape))
            else:
                batch_x = np.zeros((actual_batch_size, self.data_shape[0]), dtype=np.float32)

            for i, name in enumerate(batch_x_names):
                data, sr = self._read_audio(name)
                if self.use_mfcc:
                    batch_x[i] = self._compute_mfcc(data, sr, self.data_shape[1])
                elif self.use_spectrogram:
                    batch_x[i] = self._compute_spectrogram(data, sr)
                else:
                    batch_x[i, :data.shape[0]] = data
        else:
            batch_x = self.preprocessed_data[ids]

        if self.multi_label:
            batch_y = {f'{self.classes_names[i].replace(" ", "_")}_output': batch_y[:, i:i + 1]
                            for i in range(len(self.classes_names))}

        return batch_x, batch_y

    def on_epoch_end(self):
        self._shuffle_indices()

    def _shuffle_indices(self):
        if not self.testing:
            np.random.shuffle(self.indices)

    def _read_audio(self, file_name):
        audio_data, sr = audiofile.read(file_name)
        return audio_data, sr

    def _compute_mfcc(self, data, sr, max_frames):
        if len(data) < self.n_fft:
            data = np.pad(data, (0, self.n_fft - len(data)), mode='constant')

        mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft,
                                     hop_length=self.hop_length, win_length=self.n_fft)
        pad_width = max_frames - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        return mfccs

    def _compute_spectrogram(self, data, sr):
        if len(data) < self.n_fft:
            data = np.pad(data, (0, self.n_fft - len(data)), mode='constant')
        spectrogram = librosa.stft(data, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram = np.abs(spectrogram)
        if spectrogram.shape[1] < self.data_shape[1]:
            pad_width = self.data_shape[1] - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')

        return spectrogram

    def _multichannel_spectrogram(self,audio_data, sr, n_fft=1024, hop_length=512):
        S = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        S_delta = librosa.feature.delta(S_db)
        S_delta2 = librosa.feature.delta(S_db, order=2)

        harmonic, percussive = librosa.effects.hpss(S)
        harmonic_db = librosa.amplitude_to_db(np.abs(harmonic), ref=np.max)
        percussive_db = librosa.amplitude_to_db(np.abs(percussive), ref=np.max)
        multi_channel_spectrogram = np.stack([S_db, S_delta, S_delta2, harmonic_db, percussive_db], axis=-1)

        return multi_channel_spectrogram


    def get_label_num(self):
        """
        Assumes the y values are in the same shape as if transformed by the MultiLabelBinarizer
        from sklearn.preprocessing

        Returns: How many labels there are in a generator

        """
        return self.y.shape[1]
