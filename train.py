import tensorflow as tf


from model import BidRNNTagger
from input import Config, Inputs


def main():
    with tf.name_scope("train"):
        config = Config()
        inputs = Inputs()
        m = BidRNNTagger(config=config, inputs=inputs)

    with tf.name_scope("valid"):
        pass

    sess = tf.Session()
    init = tf.group(tf.initialize_all_variables(),
                    tf.initialize_local_variables())
    sess.run(init)
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    while not coord.should_stop():
        try:
            index = 0
        except tf.errors.OutOfRangeError:
            pass