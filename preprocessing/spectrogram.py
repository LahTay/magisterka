import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class SpectrogramGenerator:
    def __init__(self, source_dir, target_dir, image_format='png',
                 n_fft=1024, hop_length=512, n_mels=40, cmap='viridis'):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.image_format = image_format
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.cmap = cmap
        self.generate_spectrograms()

    def generate_spectrograms(self):
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory {self.source_dir} not found.")

        for batch_folder in self.source_dir.glob('batch_*'):
            print(f'Processing {batch_folder}')
            target_batch_folder = self.target_dir / batch_folder.name
            target_batch_folder.mkdir(parents=True, exist_ok=True)

            for wav_file in batch_folder.glob('*.wav'):
                spectrogram_path = target_batch_folder / (wav_file.stem + f'.{self.image_format}')
                self._create_spectrogram(wav_file, spectrogram_path)

    def _create_spectrogram(self, wav_file, target_path):
        y, sr = librosa.load(wav_file, sr=None)
        D = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        D_dB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(D_dB, sr=sr, hop_length=self.hop_length, x_axis='time', y_axis='log', cmap=self.cmap)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram of {wav_file.name}')
        plt.tight_layout()
        plt.savefig(target_path)
        plt.close()
