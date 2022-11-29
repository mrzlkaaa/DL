from __future__ import annotations
from tensorflow.keras import layers
from tensorflow.keras import losses
from .observe_data import Observe_dataset
from .preprocessing import Preprocess_text

ods = Observe_dataset()
raw_train_ds = ods.prepare_raw_ds("aclImdb/train", "training")
raw_valid_ds = ods.prepare_raw_ds("aclImdb/train", "validation")
raw_test_ds = ods.prepare_raw_ds("aclImdb/test", None)

pt = Preprocess_text()
text_train_ds, _ = pt.separate_labels_text()
pt.vl.adapt_vl(text_train_ds)

# class text_classifier:
