import pytest
import os
from src.observe_data import Observe_dataset


@pytest.fixture()
def od():
    return Observe_dataset()


def test_read_train_pos(od):
    # print(od.train_pos_path)
    od.read_pos()
    assert 0


def test_prepare_raw_ds(od):
    path, _ = os.path.split(od.train_pos_path)
    print(type(od.prepare_raw_ds(path, "training")))
    assert 0


# def test_prepare_raw_validation_ds(od):
#     path, _ = os.path.split(od.train_pos_path)
#     od.prepare_raw_training_ds(path, "validation")
#     assert 0
