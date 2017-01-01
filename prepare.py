from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import nltk
import collections
import itertools
import tensorflow as tf
import numpy as np
import sys
import cPickle as pickle
from input import VOCAB_SIZE, TAGS_SIZE
import random


def build_corpus(corpus):
    tuples = list(itertools.chain.from_iterable(corpus))
    words = [tuple[0].lower() for tuple in tuples]
    poses = [tuple[1] for tuple in tuples]
    word_counter = collections.Counter(words)
    vocab = [tuple[0] for tuple in word_counter.most_common(n=(VOCAB_SIZE - 2))]
    pos_counter = collections.Counter(poses)
    tags = [tuple[0] for tuple in pos_counter.most_common(n=(TAGS_SIZE - 2))]
    word2id = dict(zip(vocab, range(1, len(vocab)+1)))
    word2id['PAD'] = 0
    word2id['<unk>'] = VOCAB_SIZE - 1
    tag2id = dict(zip(tags, range(1, len(tags)+1)))
    tag2id['PAD'] = 0
    tag2id['<unk>'] = TAGS_SIZE - 1
    inputs = []
    labels = []
    for i, record in enumerate(corpus):
        sys.stdout.write("\r")
        sys.stdout.write(str(i))
        sys.stdout.flush()
        words_input, tags_label = zip(*record)
        words_id = [word2id[word.lower()] if word.lower() in vocab else VOCAB_SIZE - 1 for word in words_input]
        tags_id = [tag2id[tag] if tag in tags else TAGS_SIZE - 1 for tag in tags_label]
        inputs.append(words_id)
        labels.append(tags_id)
    id2word = dict([(value, key) for (key, value) in word2id.items()])
    id2tag = dict([(value, key) for (key, value) in tag2id.items()])
    print("dump dict to disk:")
    pickle.dump(word2id, open("corpus/word2id.p", "wb"))
    pickle.dump(tag2id, open("corpus/tag2id.p", "wb"))
    pickle.dump(id2word, open("corpus/id2word.p", "wb"))
    pickle.dump(id2tag, open("corpus/id2tag.p", "wb"))
    print("dump done")
    return inputs, labels


def arr2str(array):
    return np.array(array).astype(np.int64).tostring()


def convert_to_records(inputs, labels, filename):
    num_inputs = len(inputs)
    writer = tf.python_io.TFRecordWriter("records/" + filename)
    for index, (context, tags) in enumerate(zip(inputs, labels)):
        sys.stdout.write("\r")
        sys.stdout.write("write %6dth %% %d file into tfrecords file" % (index + 1, num_inputs))
        sys.stdout.flush()
        if len(context) != len(tags):
            raise ValueError("context length must be equal to that of tags")
        sequence_length = len(context)
        example = tf.train.Example(features=tf.train.Features(feature={
            "context": tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr2str(context)])),
            "tag": tf.train.Feature(bytes_list=tf.train.BytesList(value=[arr2str(tags)])),
            "sequence_length": tf.train.Feature(int64_list=tf.train.Int64List(value=[sequence_length]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    print("\nwritten finished")


def main():
    sents = list(nltk.corpus.brown.tagged_sents())
    random.shuffle(sents)
    corpus = sents
    inputs, labels = build_corpus(corpus)
    total_num = len(inputs)
    valid_num = int(0.25 * total_num)
    convert_to_records(inputs[:valid_num], labels[:valid_num], "valid.tfrecords")
    convert_to_records(inputs[valid_num:], labels[valid_num:], "train.tfrecords")