from __future__ import annotations
import tensorflow as tf
import os
import shutil


class Observe_dataset:
    SEED = 42
    BATCH_SIZE = 32

    def __init__(self):
        self.dataset_path = self.download_dataset()
        self.train_pos_path = os.path.join(self.dataset_path, "train", "pos")
        self.train_neg_path = os.path.join(self.dataset_path, "train", "neg")

    def download_dataset(self) -> str:
        url: str = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        dataset: str = tf.keras.utils.get_file(
            fname="aclImdb",
            origin=url,
            untar=True,
            cache_subdir=""
        )
        path: str = os.path.join(os.path.dirname(dataset), "aclImdb")
        # rmv_tree = os.path.join(path, "train", "unsup")
        # shutil.rmtree(rmv_tree)
        return path

    def read_pos(self):
        file_path = os.path.join(
            self.train_pos_path, os.listdir(self.train_pos_path)[0])
        with open(file_path, "r") as f:
            print(f.read())

    def prepare_raw_training_ds(self, path, subset):
        raw_ds = tf.keras.utils.text_dataset_from_directory(
            directory=path,
            batch_size=self.BATCH_SIZE,
            seed=self.SEED,
            subset=subset,
            validation_split=0.2
        )
        return raw_ds
