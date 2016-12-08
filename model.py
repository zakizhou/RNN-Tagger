from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn


class Tagger(object):
    def __init__(self, config, inputs):
        vocab_size = inputs.vocab_size
        embedding_size = config.embedding_size
        max_steps = inputs.max_steps
        num_units = config.num_units
        tags_size = inputs.tags_size
        learning_rate = config.learning_rate
        with tf.variable_scope("embedding"):
            embedding = tf.get_variable("embedding",
                                        shape=[vocab_size, embedding_size],
                                        initializer=tf.truncated_normal_initializer(stddev=0.05),
                                        dtype=tf.float32)
            look_up = tf.nn.embedding_lookup(embedding, inputs.contexts)
        squeeze = [tf.squeeze(context_, [1]) for context_ in tf.split([1], max_steps, look_up)]
        with tf.variable_scope("rnn"):
            cell = rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=True)
            outputs, _ = rnn.rnn(cell=cell,
                                 inputs=squeeze,
                                 dtype=tf.float32,
                                 sequence_length=inputs.sequence_lengths)
        with tf.variable_scope("output"):
            softmax_w = tf.get_variable("softmax_w",
                                        shape=[num_units, tags_size],
                                        initializer=tf.truncated_normal_initializer(stddev=0.05),
                                        dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b",
                                        shape=[tags_size],
                                        initializer=tf.constant_initializer(value=0.),
                                        dtype=tf.float32)
            logits = [tf.nn.xw_plus_b(output, softmax_w, softmax_b) for output in outputs]
        with tf.name_scope("loss"):
            targets = [tf.squeeze(tag_, [1]) for tag_ in tf.split([1], max_steps, inputs.tags)]
            weights_bool = [tf.greater_equal(inputs.sequence_lengths, step) for step in range(1, max_steps + 1)]
            weights = [tf.cast(weight, tf.float32) for weight in weights_bool]
            cross_entropy_per_example = tf.nn.seq2seq.sequence_loss_by_example(logits=logits,
                                                                               targets=targets,
                                                                               weights=weights)
            self.__loss = tf.reduce_mean(cross_entropy_per_example)
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.__train_op = optimizer.minimize(self.__loss)

    @property
    def loss(self):
        return self.__loss

    @property
    def train_op(self):
        return self.__train_op


class BidRNNTagger(object):
    def __init__(self, config, inputs):
        vocab_size = inputs.vocab_size
        embed_size = config.embedding_size
        forward_units = config.forward_units
        backward_units = config.backward_units
        tags_size = inputs.tags_size
        learning_rate = config.learnaing_rate
        with tf.variable_scope("embedding"):
            embed = tf.get_variable(name="embed",
                                    shape=[vocab_size, embed_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.05),
                                    dtype=tf.float32)
            look_up = tf.nn.embedding_lookup(embed, inputs.contexts)
        with tf.variable_scope("bid_rnn"):
            cell_forward = tf.nn.rnn_cell.BasicLSTMCell(num_units=forward_units, state_is_tuple=True)
            cell_backward = tf.nn.rnn_cell.BasicLSTMCell(num_units=backward_units, state_is_tuple=True)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_forward,
                                                         cell_bw=cell_backward,
                                                         inputs=look_up,
                                                         sequence_length=inputs.sequence_lengths,
                                                         dtype=tf.float32)
            # outputs[0]: [batch_size, max_steps, forward_units]
            # outputs[1]: [batch_size, max_steps, backward_units]
            # output: [batch_size, max_steps, forward_units + backward_units]
            output = tf.concat([2], outputs)
        with tf.variable_scope("output"):
            softmax_w = tf.get_variable(name="softmax_w",
                                        shape=[forward_units + backward_units, tags_size],
                                        initializer=tf.truncated_normal_initializer(stddev=0.05),
                                        dtype=tf.float32)
            softmax_b = tf.get_variable(name="softmax_b",
                                        shape=[tags_size],
                                        initializer=tf.constant_initializer(value=0.),
                                        dtype=tf.float32)
            logits = tf.add(tf.matmul(output, softmax_w), softmax_b)
        with tf.name_scope("loss"):
            fake_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=inputs.tags)
            mask = tf.sequence_mask(inputs.sequence_lengths, dtype=tf.float32)
            loss_per_example_per_step = tf.mul(fake_loss, mask)
            loss_per_example_sum = tf.reduce_sum(loss_per_example_per_step, reduction_indices=[2])
            loss_per_example_average = tf.div(loss_per_example_sum, inputs.sequence_lengths)
            self.__loss = tf.reduce_mean(loss_per_example_average)
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.__train_op = optimizer.minimize(self.__loss)

    @property
    def loss(self):
        return self.__loss

    @property
    def train_op(self):
        return self.__train_op
