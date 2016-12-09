import tensorflow as tf


from model import BidRNNTagger
from input import Config, Inputs


def main():
    with tf.variable_scope("model", reuse=None):
        with tf.name_scope("train"):
            inputs = Inputs(train=True)
            config = Config(train=True)
            m = BidRNNTagger(config=config, inputs=inputs)

    with tf.variable_scope("model", reuse=True):
        with tf.name_scope("valid"):
            valid_inputs = Inputs(train=False)
            valid_config = Config(train=True)
            mvalid = BidRNNTagger(config=valid_config, inputs=valid_inputs)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.group(tf.initialize_all_variables(),
                    tf.initialize_local_variables())
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        index = 1
        while not coord.should_stop():
            _, train_loss = sess.run([m.train_op, m.loss])
            print("step: " + str(index) + " loss:" + str(train_loss))
            if (index + 1) % 5 == 0:
                valid_accuracy = sess.run(mvalid.validate)
                print("valid loss now: " + str(valid_accuracy))
            index += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
        del sess
    except KeyboardInterrupt:
        print("detect keyboard interrupt, stopping!")
        del sess
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    del sess