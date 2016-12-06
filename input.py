from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import nltk
import collections
import itertools
import tensorflow as tf
import numpy as np
import sys


VOCAB_SIZE = 50000
TAGS_SIZE = 350
BATCH_SIZE = 3
CAPACITY = 100


def build_corpus(corpus):
    tuples = list(itertools.chain.from_iterable(corpus))
    words = [tuple[0].lower() for tuple in tuples]
    poses = [tuple[1] for tuple in tuples]
    word_counter = collections.Counter(words)
    vocab = [tuple[0] for tuple in word_counter.most_common(n=VOCAB_SIZE)]
    pos_counter = collections.Counter(poses)
    tags = [tuple[0] for tuple in pos_counter.most_common(n=TAGS_SIZE)]
    word2id = dict(zip(vocab, range(1, len(vocab)+1)))
    word2id['PAD'] = 0
    tag2id = dict(zip(tags, range(1, len(tags)+1)))
    tag2id['PAD'] = 0
    inputs = []
    labels = []
    for record in corpus:
        words_input, tags_label = zip(*record)
        words_id = [word2id[word.lower()] for word in words_input]
        tags_id = [tag2id[tag] for tag in tags_label]
        inputs.append(words_id)
        labels.append(tags_id)
    return word2id, tag2id, inputs, labels


def arr2str(array):
    return np.array(array).astype(np.int64).tostring()


def convert_to_records(inputs, labels):
    num_inputs = len(inputs)
    writer = tf.python_io.TFRecordWriter("records/examples.tfrecords")
    for index, (context, tags) in enumerate(zip(inputs, labels)):
        sys.stdout.write("\r")
        sys.stdout.write("write %6dth %% %d file into tfrecords file" % (index + 1, num_inputs))
        sys.stdout.flush()
        if len(context) != len(tags):
            raise ValueError("context length must be equal to that of tags")
        sequence_length = len(context)
        example = tf.train.Example(features=tf.train.Features(feature={
            "context": tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr2str(context)])),
            "tags": tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr2str(tags)])),
            "sequence_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence_length]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print("\nwritten finished")


corpus = nltk.corpus.brown.tagged_sents()[:1000]
_, _, inputs, labels = build_corpus(corpus)
convert_to_records(inputs, labels)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(serialized,
                                       features={
                                           "context": tf.FixedLenFeature([], dtype=tf.string),
                                           "tags": tf.FixedLenFeature([], dtype=tf.string),
                                           "sequence_length": tf.FixedLenFeature([], dtype=tf.int64)
                                       })
    sequence_length = features['sequence_length']
    context = tf.decode_raw(features['context'], tf.int64)
    tag = tf.decode_raw(features['tags'], tf.int64)
    return context, tag, sequence_length


def input_producer(context, tag, sequence_length):
    contexts, tags, sequence_lengths = tf.train.batch([context, tag, sequence_length],
                                                      batch_size=BATCH_SIZE,
                                                      capacity=CAPACITY,
                                                      dynamic_pad=True)
    return contexts, tags, sequence_lengths

filename_queue = tf.train.string_input_producer(["records/examples.tfrecords"])
context, tag, sequence_length = read_and_decode(filename_queue)


class Inputs(object):
    def __init__(self):
        self.max_steps = 50
        self.vocab_size = VOCAB_SIZE
        self.tags_size = TAGS_SIZE
        self.contexts, self.tags, self.sequence_lengths = input_producer(context, tag, sequence_length)


class Config(object):
    def __init__(self):
        self.embedding_size = 256
        self.num_units = 54
        self.learnaing_rate = 5e-4
