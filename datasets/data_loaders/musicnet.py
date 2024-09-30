import csv

import os
import os.path
import pickle
import shutil

from intervaltree import IntervalTree
from scipy.io import wavfile


class MusicNet:
    """`MusicNet <https://zenodo.org/records/5120004/files/musicnet.tar.gz>`_ Dataset.
    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from ``train_data``,
            otherwise from ``test_data``.
        preprocess (bool, optional): If true, preprocesses the dataset to a necessary form
            puts it in root directory. Should be done once and at first run of the class.
    """

    """
    BEFORE USING - you have to merge your train and test data into 2 folders (data and labels)
    Main usage:
    Set up the root folder, train and preprocess variables.
    Then call the instance to receive data and labels.
    """

    def __init__(self, root, preprocess=False):
        self.root = os.path.expanduser(root)

        self.data_folder = 'data'
        self.labels_folder = "labels"
        self.tree_file1 = "train_tree.pckl"
        self.tree_file2 = "test_tree.pckl"

        if preprocess:
            self.preprocess()

        if not self._check_exists():
            raise RuntimeError(f'Dataset not found in {self.root}.\n Run the class with preprocessing flag on.')

        self.data_path = os.path.join(self.root, self.data_folder)
        labels_path = os.path.join(self.root, self.labels_folder, self.tree_file1)
        labels_path2 = os.path.join(self.root, self.labels_folder, self.tree_file2)

        print(f"Reading label pickle data...")

        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)


        with open(labels_path2, 'rb') as f:
            labels2 = pickle.load(f)
        self.labels.update(labels2)

        self.rec_ids = list(self.labels.keys())
        self.data_files = [os.path.join(self.data_path, f"{rec_id}.wav") for rec_id in self.rec_ids]

    def get_names_list(self):
        return self.data_files

    def get_labels_obj(self):
        return self.labels

    def __call__(self, *args, **kwargs):
        return self.get_names_list(), self.get_labels_obj()

    def _check_exists(self):
        train_data_exist = os.path.exists(os.path.join(self.root, self.data_folder))
        test_data_exist = os.path.exists(os.path.join(self.root, self.labels_folder))

        return train_data_exist and test_data_exist

    def preprocess(self):
        """
        Take the raw downloaded data from the self.root folder.
        If it's compressed, uncompress it.
        If it's in .tar restore it into folders.
        Preprocess the csv label data into a pickle file.

        :return: True if preprocessing ended positively, False if it's not necessary
        """

        import tarfile
        import os
        from pathlib import Path

        if self._check_exists():
            return False

        filenames = os.listdir(self.root)

        if not filenames:
            raise Exception("The root folder is empty")

        # Check if there exist already extracted folders if not extract the tar.gz file
        if not any([os.path.isfile(os.path.join(self.root, file)) and file.endswith(".gz") for file in filenames]):
            for filename in filenames:
                if filename.endswith('.gz'):
                    gz_path = os.path.join(self.root, filename)
                    with tarfile.open(gz_path, 'r:gz') as tar:
                        tar.extractall(path=self.root)

                    print(f"Extracted {filename} in {self.root}")

        dir_count = sum(os.path.isdir(os.path.join(self.root, file)) for file in filenames)

        # Move the folders one up, so they are in the root folder
        if dir_count == 1:
            for filename in filenames:
                p = Path(self.root).absolute() / filename
                if p.is_dir() and p.name == "musicnet":
                    for item in p.iterdir():
                        item.rename(p.parent / item.name)
                    os.rmdir(p)

        trees = self.process_labels(self.test_labels)
        with open(os.path.join(self.root, self.test_labels, self.test_tree), 'wb') as f:
            pickle.dump(trees, f)

        trees = self.process_labels(self.train_labels)
        with open(os.path.join(self.root, self.train_labels, self.train_tree), 'wb') as f:
            pickle.dump(trees, f)

        self.refresh_cache = False
        print('Preprocessing Complete')
        return True

    # write out wavfiles as arrays for direct mmap access
    def process_data(self, path):
        for item in os.listdir(os.path.join(self.root, path)):
            if not item.endswith('.wav'): continue
            uid = int(item[:-4])
            _, data = wavfile.read(os.path.join(self.root, path, item))
            data.tofile(os.path.join(self.root, path, item[:-4] + '.bin'))

    # wite out labels in intervaltrees for fast access
    def process_labels(self, path):
        trees = dict()
        for item in os.listdir(os.path.join(self.root, path)):
            if not item.endswith('.csv'): continue
            uid = int(item[:-4])
            tree = IntervalTree()
            with open(os.path.join(self.root, path, item), 'r') as f:
                reader = csv.DictReader(f, delimiter=',')
                for label in reader:
                    start_time = int(label['start_time'])
                    end_time = int(label['end_time'])
                    instrument = int(label['instrument'])
                    tree[start_time:end_time] = instrument
            trees[uid] = tree
        return trees
