import tensorflow as tf


from model import BidRNNTagger
from input import Config, Inputs

LEARNING_RATE = 4e-5
NUM_GPUS = 2


def averaged_gradients(tower_gradients):
    gradients_mean = []
    for gradient_variable in zip(*tower_gradients):
        gradients = []
        for gradient, _ in gradient_variable:
            expanded_gradient = tf.expand_dims(gradient, 0)
            gradients.append(expanded_gradient)
        concat = tf.concat(0, gradients)
        averaged_gradient = tf.reduce_mean(concat, reduction_indices=0)
        grad_var_tuple = (averaged_gradient, gradient_variable[0][1])
        gradients_mean.append(grad_var_tuple)
    return gradients_mean


def main():
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        gradients = []
        losses = []
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        for gpu_index in range(NUM_GPUS):
            gpu_name = "/gpu:%d" % gpu_index
            with tf.device(gpu_name):
                with tf.name_scope("tower_%d" % gpu_index):
                    inputs = Inputs(train=True)
                    config = Config(train=True)
                    m = BidRNNTagger(config=config, inputs=inputs)
                    gradient = optimizer.compute_gradients(m.loss)
                    losses.append(m.loss)
                    tf.get_variable_scope().reuse_variables()
                    gradients.append(gradient)
        gradients_mean_across_gpus = averaged_gradients(gradients)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients_mean_across_gpus)
        loss_mean_across_gpus = tf.reduce_mean(tf.pack(losses))
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
                _, train_loss = sess.run([train_op, loss_mean_across_gpus])
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