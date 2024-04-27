from pathlib import Path
import pandas as pd
import math
from tqdm import tqdm

def load_cut_files(dataset_path):
    """

    Args:
        dataset_path (str | Path): Path to the cut training dataset

    Returns:

    """

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise Exception("Given folders do not exists.")

    csv_path = dataset_path / "segments_instruments.csv"
    if not csv_path.exists():
        raise Exception("CSV file does not exist in the given dataset path.")

    labels = pd.read_csv(csv_path)

    filenames = []
    returned_labels = []

    for index, row in tqdm(labels.iterrows(), total=labels.shape[0], desc="Loading dataset"):
        filename = row["Filename"]
        instruments = row["Instruments"]
        batch_id = row["BatchID"]

        full_path = dataset_path / batch_id / filename

        if isinstance(instruments, float) and math.isnan(instruments):
            instruments = []
        else:
            instruments = [int(instrument) for instrument in str(instruments).split(",")]

        filenames.append(full_path)
        returned_labels.append(instruments)

    return filenames, returned_labels
