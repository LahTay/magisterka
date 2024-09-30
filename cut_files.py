from datasets.cut_dataset.cut_audio_file import main_audio_cutter
from pathlib import Path


def file_cutter(root_path, dataset, segment_length, overlap_length):
    cut_files_folder_name = f"{segment_length}s_len.{overlap_length}s_overlap"
    main_audio_cutter(root_path, cut_files_folder_name, segment_length, overlap_length, dataset, None)

if __name__ == "__main__":

    cutting_files = False

    segment_length = 0.25
    overlap_length = 0

    root_path = Path("./datasets/datasets/musicnet").resolve()
    cut_files_folder_name = f"{segment_length}s_len.{overlap_length}s_overlap"

    root_path_pretraining = Path("./datasets/datasets/musan").resolve()

    if cutting_files:
        file_cutter(root_path, "MusicNet", segment_length, overlap_length)
        file_cutter(root_path_pretraining, "Musan", segment_length, overlap_length)

