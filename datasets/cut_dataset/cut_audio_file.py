from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa
import soundfile as sf
from intervaltree import IntervalTree

from datasets.data_loaders.musicnet import MusicNet
from datasets.data_loaders.musan import MusanDataset


def slice_audio_file(filename, interval_tree, segment_length_sec, overlap_length_sec, sr=None):
    """
    Slices an audio file into overlapping segments and assigns labels based on an IntervalTree.

    Args:
        filename (str): The path to the audio file.
        interval_tree (IntervalTree): An IntervalTree mapping intervals of sample indices to instrument IDs.
        segment_length_sec (int | float): The length of each segment in seconds.
        overlap_length_sec (int | float): The overlap between consecutive segments in seconds.
        sr (int | None, optional): The sampling rate to which the audio should be resampled. If None, the native sampling rate is used.

    Returns:
        list of tuples: Each tuple contains an audio segment as an ndarray and a set of instrument labels active in that segment.
    """

    # Load the audio file
    audio, sr = librosa.load(filename, sr=sr)  # Load with the native sampling rate

    # Calculate segment and overlap length in samples
    segment_length_samples = int(segment_length_sec * sr)
    overlap_length_samples = int(overlap_length_sec * sr)

    # Calculate the number of segments to create
    num_segments = int(np.ceil((len(audio) - overlap_length_samples) / (segment_length_samples - overlap_length_samples)))

    # List to hold the resulting slices and their labels
    sliced_audio_and_labels = []

    for i in range(num_segments):
        # Calculate the start and end indices of the segment
        start_sample = i * (segment_length_samples - overlap_length_samples)
        end_sample = start_sample + segment_length_samples
        end_sample = min(end_sample, len(audio))  # Ensure we don't go past the audio length

        # Extract the segment
        audio_segment = audio[start_sample:end_sample]

        # Find matching intervals in the IntervalTree
        matching_intervals = interval_tree.overlap(start_sample, end_sample)
        instruments_in_segment = {interval.data for interval in matching_intervals}

        # Append the results
        sliced_audio_and_labels.append((audio_segment, instruments_in_segment))

    return sliced_audio_and_labels, sr

def slice_audio_musan(filename, label, segment_length_sec, overlap_length_sec, sr=None):
    audio, sr = librosa.load(filename, sr=sr)  # Load with the native sampling rate
    # Calculate segment and overlap length in samples
    segment_length_samples = int(segment_length_sec * sr)
    overlap_length_samples = int(overlap_length_sec * sr)

    # Calculate the number of segments to create
    num_segments = int(
        np.ceil((len(audio) - overlap_length_samples) / (segment_length_samples - overlap_length_samples)))

    # List to hold the resulting slices and their labels
    sliced_audio_and_labels = []

    for i in range(num_segments):
        # Calculate the start and end indices of the segment
        start_sample = i * (segment_length_samples - overlap_length_samples)
        end_sample = start_sample + segment_length_samples
        end_sample = min(end_sample, len(audio))  # Ensure we don't go past the audio length

        # Extract the segment
        audio_segment = audio[start_sample:end_sample]

        # Append the results
        sliced_audio_and_labels.append((audio_segment, label))

    return sliced_audio_and_labels, sr



def clear_folder(folder_path):
    """
    Deletes all contents of a folder.

    Args:
        folder_path (Path or str): The folder whose contents are to be deleted.
    """
    folder_path = Path(folder_path)
    for item in folder_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def process_musicnet(wav_files, labels, output_folder, segment_length_sec, overlap_length_sec, sr_base):
    csv_data = []
    csv_path = output_folder / "segments_instruments.csv"
    file_count = 0
    batch_index = 0
    batch_folder = output_folder / f"batch_{batch_index}"
    batch_folder.mkdir(parents=True, exist_ok=True)

    for wav_file in tqdm(wav_files, desc="Processing WAV files"):
        filename_id = int(Path(wav_file).stem)
        interval_tree = labels[filename_id]
        segments, sr = slice_audio_file(wav_file, interval_tree, segment_length_sec, overlap_length_sec, sr_base)
        filename_id = Path(wav_file).stem

        for i, (segment, instruments) in enumerate(segments):

            if file_count >= 1000:
                file_count = 0
                batch_index += 1
                batch_folder = output_folder / f"batch_{batch_index}"
                batch_folder.mkdir(parents=True, exist_ok=True)

            segment_filename = f"{filename_id}_{i}.wav"
            segment_path = batch_folder / segment_filename
            sf.write(segment_path, segment, int(sr))

            # Convert set of instruments to a comma-separated string
            instruments_str = ','.join(map(str, instruments))
            csv_data.append([segment_filename, instruments_str, f"batch_{batch_index}"])
            file_count += 1
    # Save the CSV file
    df = pd.DataFrame(csv_data, columns=["Filename", "Instruments", "BatchID"])
    df.to_csv(csv_path, index=False)

def process_musan(wav_files, labels, output_folder, segment_length_sec, overlap_length_sec, sr_base):
    csv_data = []
    csv_path = output_folder / "segments_instruments.csv"
    file_count = 0
    batch_index = 0
    batch_folder = output_folder / f"batch_{batch_index}"
    batch_folder.mkdir(parents=True, exist_ok=True)
    for wav_file, label in tqdm(zip(wav_files, labels), desc="Processing WAV files", total=len(wav_files)):
        segments, sr = slice_audio_musan(wav_file, label, segment_length_sec, overlap_length_sec, sr_base)
        filename_id = Path(wav_file).stem
        for i, (segment, genre) in enumerate(segments):
            if file_count >= 1000:
                file_count = 0
                batch_index += 1
                batch_folder = output_folder / f"batch_{batch_index}"
                batch_folder.mkdir(parents=True, exist_ok=True)

            segment_filename = f"{filename_id}_{i}.wav"
            segment_path = batch_folder / segment_filename
            sf.write(segment_path, segment, int(sr))

            # Convert set of genre to a comma-separated string
            instruments_str = ','.join(map(str, genre))
            csv_data.append([segment_filename, instruments_str, f"batch_{batch_index}"])
            file_count += 1

    df = pd.DataFrame(csv_data, columns=["Filename", "Instruments", "BatchID"])
    df.to_csv(csv_path, index=False)


def cut_audio_and_save(wav_files, labels, output_folder, segment_length_sec, overlap_length_sec, dataset,
                       sr_base=None):
    """
    Processes a list of WAV files, slicing them into segments and saving both the segments
    and a CSV file with segment names and corresponding instruments.

    Args:
        wav_files (list): List of paths to WAV files.
        labels (dict): Dictionary mapping WAV file names to their corresponding IntervalTree.
        output_folder (str or Path): Folder where the sliced WAV files and CSV will be saved.
        segment_length_sec (int | float): Length of each segment in seconds.
        overlap_length_sec (int | float): Overlap between segments in seconds.
        dataset (str): Either MusicNet or Musan
        sr_base (int | None): Sampling rate to resample the audio. If None, the native sampling rate is used.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    clear_folder(output_folder)

    if dataset == "MusicNet":
        process_musicnet(wav_files, labels, output_folder, segment_length_sec, overlap_length_sec, sr_base)
    elif dataset == "Musan":
        process_musan(wav_files, labels, output_folder, segment_length_sec, overlap_length_sec, sr_base)
    else:
        raise Exception(f"Wrong dataset name {dataset}")




def main_audio_cutter(root_directory, cut_files_folder_name, segment_length, overlap_length, dataset, sr_base=None):
    """
    Processes a list of WAV files, slicing them into segments and saving both the segments
    and a CSV file with segment names and corresponding instruments.

    Args:
        root_directory (str | Path): Root directory of the MusicNet dataset.
        cut_files_folder_name (str | path): Name of the folder for the output cut files.
        segment_length (int | float): Length of each segment in seconds.
        overlap_length (int | float): Overlap between segments in seconds.
        sr_base (int | None): Sampling rate to resample the audio. If None, the native sampling rate is used.
    """

    if dataset == 'MusicNet':
        data, labels = MusicNet(root_directory)()
    elif dataset == 'Musan':
        musan_dataset = MusanDataset(root_directory)()
        data = musan_dataset.index.tolist()
        labels = musan_dataset['genre'].apply(lambda x: x.split(',')).tolist()
    else:
        raise Exception(f"This dataset doesn't exist: {dataset}")

    data_path = Path(root_directory).absolute() / cut_files_folder_name

    cut_audio_and_save(data, labels, data_path, segment_length, overlap_length, dataset, sr_base)
