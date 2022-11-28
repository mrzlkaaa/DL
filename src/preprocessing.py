import tensorflow as tf
from tensorflow.keras import layers
import re
import string


class Preprocess_text:
    max_features = 20000
    sequence_length = 250

    def __init__(self, ds):
        self.ds = ds
        self.data = self.ds.map(lambda x, _: x)
        self.labels = self.ds.map(lambda _, y: y)

    # * remove all html tagges, spaces and punctuation

    def custom_html_standartization(self, input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')

    def vectoriztion_layer(self, data):
        vectorize_layer = layers.TextVectorization(
            standardize=self.custom_html_standartization,
            max_tokens=self.max_features,
            output_mode='int',
            output_sequence_length=self.sequence_length

        )
        return vectorize_layer

    def vectorize_and_adapt(self):
        return self.vectorize_layer.adapt(self.data)

    def vectorize(self):
        # return print(self.vectorize_layer(self.data))
        for t_batch, _ in self.ds:
            t = t_batch[0]
            t = tf.expand_dims(t, -1)
            self.vectorize_layer(t)
            print(t)

            # print(self.vectorize_layer(t))

        # return data
