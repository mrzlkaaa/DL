import pytest
from src.preprocessing import Preprocess_text, custom_html_standartization
from src.observe_data import Observe_dataset

txt_1 = '"Pandemonium" is a horror movie spoof that comes off more stupid than funny. Believe me when I tell you, I love comedies. Especially comedy spoofs. \
"Airplane", "The Naked Gun" trilogy, "Blazing Saddles", "High Anxiety", and "Spaceballs" are some of my favorite comedies that spoof a particular genre. \
"Pandemonium" is not up there with those films. Most of the scenes in this movie had me sitting there in stunned silence because the movie wasn\'t all that \
funny. There are a few laughs in the film, but when you watch a comedy, you expect to laugh a lot more than a few times and that\'s all this film has going \
for it. Geez, "Scream" had more laughs than this film and that was more of a horror film. How bizarre is that?<br /><br />*1/2 (out of four)'

raw_train_ds = Observe_dataset().prepare_raw_ds(
        # "aclImdb/train",
        "/home/mrzlka/.keras/aclImdb",
        "training"
        )
raw_valid_ds = Observe_dataset().prepare_raw_ds(
        # "aclImdb/train",
        "/home/mrzlka/.keras/aclImdb",
        "validation"
        )


@pytest.fixture
def pt():
    return Preprocess_text()

def test_standartization_1():
    print(custom_html_standartization(txt_1))
    assert 0

def test_standartization(pt):
    print(raw_train_ds)
    text, label = pt.separate_labels_text(raw_train_ds)
    for i in text:
        pt.custom_html_standartization(i)
    assert 0

def test_vectorize(pt):
    text, _ = pt.separate_labels_text(raw_train_ds)
    pt.vl.adapt(text)
    pt.vl.get_vocabulary()
    text_valid, _ = pt.separate_labels_text(raw_train_ds)
    validation_ds = text_valid.map(lambda x: pt.vl(x))
    for i in validation_ds.take(2):
        print(i)
    assert 0


