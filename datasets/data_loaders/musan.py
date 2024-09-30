from pathlib import Path
import pandas as pd


# This dataset doesn't contain labels of instruments in the data.
# That is why it returns only a list of wav files for unsupervised training
class MusanDataset:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.music_folder = self.base_path / 'music'
        self.dataframe = self._load_annotations()

    def _load_annotations(self):
        """Load annotations for all files in the music directory."""
        annotations = []
        groups = list(self.music_folder.iterdir())

        for group in groups:
            annotation_file = group / 'ANNOTATIONS'

            if annotation_file.exists():


                # For some reason one file has 6 columns even though it doesn't really have them?
                # So for error is just so it can get filtered and go through
                df = pd.read_csv(
                    annotation_file,
                    delimiter=' ',
                    names=['file', 'genre', 'vocals', 'artist', 'composer', 'for_error'],
                    converters={'vocals': lambda x: x == 'Y'},
                    index_col='file',
                    header=None,
                    engine='python',

                )

                df.index = f'{group.name}/' + df.index + '.wav'  # Adjust file paths
                annotations.append(df)

        # Combine all annotations into a single DataFrame
        annotations_df = pd.concat(annotations, ignore_index=False)
        annotations_df = annotations_df.drop(columns=['for_error'])
        annotations_df.index = [
            (self.music_folder / file_path).resolve() for file_path in annotations_df.index
        ]
        return annotations_df


    def __call__(self):
        """Return: A pandas dataframe where the index is the filename nad the rest are its annotations"""
        return self.dataframe
