import math
import numpy as np
from tensorflow.keras.utils import Sequence
import audiofile
import librosa

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
    def __init__(self, x_paths, y_values, batch_size=64, use_mfcc=False, n_fft=1024, hop_length=512, n_mfcc=40):
        self.x = x_paths  # Has to be ndarray
        self.y = y_values  # Has to be ndarray
        self.batch_size = batch_size
        self.use_mfcc = use_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.indices = np.arange(x_paths.shape[0])
        self._shuffle_indices()
        self.len = math.ceil(self.x.shape[0] / self.batch_size)

        if use_mfcc:
            # Test read an audio file to set input dimension
            sample_data, sr = librosa.load(self.x[0], sr=None)
            # Use librosa to calculate a spectrogram shape as a test to set input dimensions
            mfccs = librosa.feature.mfcc(y=sample_data, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft,
                                         hop_length=self.hop_length, win_length=self.n_fft, window='hamming')
            self.data_shape_mfcc = mfccs.shape

        sample_data, _ = audiofile.read(self.x[0])
        self.data_shape = sample_data.shape

        print(f"Generator initialized with {len(self.x)} audio files.")
        print(f"Total batches per epoch: {self.len}")
        print(f"Audio data has a shape of {self.data_shape}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ids = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]


        actual_batch_size = len(ids)
        #if actual_batch_size != self.batch_size:
            #print("It's here!!!!!!!!!")

        batch_x_names = self.x[ids]

        if self.use_mfcc:
            batch_x = np.zeros((actual_batch_size, *self.data_shape_mfcc))
        else:
            batch_x = np.zeros((actual_batch_size, self.data_shape[0]), dtype=np.float32)
        batch_y = self.y[ids]

        for i, name in enumerate(batch_x_names):
            data, sr = self._read_audio(name)
            if self.use_mfcc:
                batch_x[i] = self._compute_mfcc(data, sr, self.data_shape_mfcc[1])
            else:
                batch_x[i, :data.shape[0]] = data



        #print(f"  ----  Batch {idx} shapes - X: {batch_x.shape}, Y: {batch_y.shape}")
        return batch_x, batch_y

    def on_epoch_end(self):
        self._shuffle_indices()

    def _shuffle_indices(self):
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
