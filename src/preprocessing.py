import tensorflow as tf
from tensorflow.keras import layers
import re
import string


def custom_html_standartization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


class Preprocess_text:
    max_features = 20000
    sequence_length = 250

    def __init__(self):
        self.vl = self.vectorization_layer()

    def separate_labels_text(self, dataset):
        text = dataset.map(lambda x, _: x)
        label = dataset.map(lambda _, y: y)
        return text, label

    # * remove all html tagges, spaces and punctuations
    def custom_html_standartization(self, input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')

    def vectorization_layer(self):
        return layers.TextVectorization(
            standardize=self.custom_html_standartization,
            max_tokens=self.max_features,
            output_mode='int',
            output_sequence_length=self.sequence_length
        )

    def adapt_vl(self, data):
        return self.vl.adapt(data)

        # print(self.vectorize_layer(t))

        # return data
