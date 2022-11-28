import pytest
from src.preprocessing import Preprocess_text
from src.observe_data import Observe_dataset


@pytest.fixture
def pt_train():
    train_ds = Observe_dataset().prepare_raw_training_ds(
        "aclImdb/train",
        "training"
    )
    return Preprocess_text(train_ds)


def test_standartization(pt_train):
    pt_train.custom_html_standartization()
    assert 0


def test_vectorize(pt_train):
    # ts = pt_train.vectorize()
    # print(next(iter(pt_train.data)))
    pt_train.vectorize()
    assert 0
