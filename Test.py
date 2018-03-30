""" Testing the network on a single GPU """

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

import time, os, glob

import HeatMatrix as network
import numpy as np
import tensorflow as tf
import SODTester as SDT
import tensorflow.contrib.slim as slim
import SODLoader as SDL

sdl = SDL.SODLoader(data_root='data/')

_author_ = 'Simi'

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Group 1: 3796, Group 2 3893
# 2 class: 2547 and 2457
tf.app.flags.DEFINE_integer('epoch_size', 370, """Test examples: OF: 508""")
tf.app.flags.DEFINE_integer('batch_size', 74, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_classes', 2, """ Number of classes""")
tf.app.flags.DEFINE_string('test_files', '0', """Files for testing have this name""")
tf.app.flags.DEFINE_integer('box_dims', 512, """dimensions of the input pictures""")
tf.app.flags.DEFINE_integer('network_dims', 256, """the dimensions fed into the network""")

# Hyperparameters:
tf.app.flags.DEFINE_float('dropout_factor', 1.0, """ p value for the dropout layer""")
tf.app.flags.DEFINE_float('l2_gamma', 1e-4, """ The gamma value for regularization loss""")
tf.app.flags.DEFINE_float('loss_factor', 1.0, """Penalty for missing a class is this times more severe""")
tf.app.flags.DEFINE_float('moving_avg_decay', 0.999, """ The decay rate for the moving average tracker""")

# Directory control
tf.app.flags.DEFINE_string('train_dir', 'training/', """Directory to write event logs and save checkpoint files""")
tf.app.flags.DEFINE_string('RunInfo', 'Extended/', """Unique file name for this training run""")
tf.app.flags.DEFINE_integer('GPU', 0, """Which GPU to use""")


# Define a custom training class
def eval():

    # Makes this the default graph where all ops will be added
    with tf.Graph().as_default(), tf.device('cpu:0'):

        # Get a dictionary of our images, id's, and labels here. Use the CPU
        _, valid = network.inputs(skip=True)

        # Define phase of training
        phase_train = tf.placeholder(tf.bool)

        # Build a graph that computes the prediction from the inference model (Forward pass)
        logits, _ = network.forward_pass_extend(valid['data'], phase_train=phase_train)
        # logits = tf.nn.softmax(logits)

        # To retreive labels
        labels = valid['label_data']

        # Initialize variables operation
        var_init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Restore moving average of the variables
        var_ema = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay)

        # Define variables to restore
        var_restore = var_ema.variables_to_restore()

        # Initialize the saver
        saver = tf.train.Saver(var_restore, max_to_keep=3)

        # Trackers for best performers
        best_MAE, best_epoch = 0, 0

        while True:

            # Allow memory placement growth
            config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:

                # Retreive the checkpoint
                ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir + FLAGS.RunInfo)

                # Initialize the variables
                sess.run(var_init)

                if ckpt and ckpt.model_checkpoint_path:

                    # Restore the learned variables
                    restorer = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

                    # Restore the graph
                    restorer.restore(sess, ckpt.model_checkpoint_path)

                    # Extract the epoch
                    Epoch = ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1]

                # Set the max step count
                max_steps = int(FLAGS.epoch_size / FLAGS.batch_size)

                # Define tester class instance
                sdt = SDT.SODTester(False, False)
                display_lab, display_log, display_img = [], [], []

                # Use slim to handle queues:
                with slim.queues.QueueRunners(sess):

                    for i in range(max_steps):

                        # Load some metrics for testing
                        lbl1, logtz, imgz, serz = sess.run([labels, logits, valid['data'], valid['patient']], feed_dict={phase_train: False})

                        # Combine predictions
                        if i == 0: display_lab, display_log, display_img = lbl1, logtz, imgz
                        else: display_lab, display_log, display_img = np.concatenate((display_lab, lbl1)), np.concatenate((display_log, logtz)), np.concatenate((display_img, imgz))
                        print(display_lab.shape, display_log.shape, display_img.shape)

                    # Retreive metrics
                    label_track, logit_track, img_track = np.squeeze(display_lab), np.squeeze(display_log[:,:,:,(FLAGS.num_classes-1)]), np.squeeze(display_img)
                    print ("Number of Examples: ", label_track.shape, logit_track.shape, img_track.shape)

                    # Normalize display
                    # label_track[label_track > 0] = 1
                    # logit_track *= label_track
                    # for v in range (logit_track.shape[0]):
                    #     logit_track[v] = (logit_track[v] - np.mean(logit_track[v]))/np.std(logit_track[v])
                    #     logit_track[v][label_track[v]==0] == np.min(logit_track[v])

                    # Display volume
                    sdl.display_volume(logit_track, True, cmap='jet')
                    # sdl.display_volume(img_track, True)


def main(argv=None):  # pylint: disable=unused-argument
    time.sleep(0)
    if tf.gfile.Exists('testing/' + FLAGS.RunInfo):
        tf.gfile.DeleteRecursively('testing/' + FLAGS.RunInfo)
    tf.gfile.MakeDirs('testing/' + FLAGS.RunInfo)
    eval()


if __name__ == '__main__':
    tf.app.run()