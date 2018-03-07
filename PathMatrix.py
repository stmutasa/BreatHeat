# Defines and builds our network
#    Computes input images and labels using inputs() or distorted inputs ()
#    Computes inference on the models (forward pass) using inference()
#    Computes the total loss using loss()
#    Performs the backprop using train()

from __future__ import absolute_import  # import multi line and Absolute/Relative
from __future__ import division  # change the division operator to output float if dividing two integers
from __future__ import print_function  # use the print function from python 3

_author_ = 'simi'

import tensorflow as tf
import Input
import SODNetwork as SDN

# Define the FLAGS class to hold our immutable global variables
FLAGS = tf.app.flags.FLAGS

# Define some of the immutable variables
tf.app.flags.DEFINE_string('data_dir', 'data/', """Path to the data directory.""")

# Retreive helper function object
sdn = SDN.SODMatrix()


def forward_pass(images, phase_train):

    """
    Perform forward pass
    :param images: 3 channel images
    :param phase_train: training or testing phase
    :return: logits, l2 loss and last conv layaer
    """

    # First layer is conv
    conv = sdn.convolution('Conv1', images, 3, 16, 1, phase_train=phase_train)
    print('Input Images: ', images)

    conv = sdn.residual_layer('Residual1', conv, 3, 32, 2, phase_train=phase_train)
    conv = sdn.residual_layer('Residual2', conv, 3, 64, 2, phase_train=phase_train)
    conv = sdn.residual_layer('Residual3', conv, 3, 128, 2, phase_train=phase_train)
    conv = sdn.wide_residual_layer('Residual3a', conv, 128, 1, phase_train=phase_train)
    conv = sdn.wide_residual_layer('Residual4', conv, 256, 2, phase_train=phase_train)
    conv = sdn.wide_residual_layer('Residual4a', conv, 256, 1, phase_train=phase_train)
    conv = sdn.wide_residual_layer('Residual4b', conv, 256, 1, phase_train=phase_train)
    conv = sdn.wide_residual_layer('Residual5', conv, 512, 2, phase_train=phase_train)
    conv = sdn.wide_residual_layer('Residual5a', conv, 512, 1, phase_train=phase_train)
    conv = sdn.wide_residual_layer('Residual5b', conv, 512, 1, phase_train=phase_train)
    conv = sdn.wide_residual_layer('Residual5c', conv, 512, 1, phase_train=phase_train)

    # Residual blocks
    # conv = sdn.residual_layer('Residual1', conv, 3, 32, 2, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual1a', conv, 3, 32, 1, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual2', conv, 3, 64, 2, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual2a', conv, 3, 64, 1, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual3', conv, 3, 128, 2, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual3a', conv, 3, 128, 1, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual4', conv, 3, 256, 2, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual4a', conv, 3, 256, 1, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual4b', conv, 3, 256, 1, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual5', conv, 3, 512, 2, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual5a', conv, 3, 512, 1, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual5b', conv, 3, 512, 1, phase_train=phase_train)
    # conv = sdn.residual_layer('Residual5c', conv, 3, 512, 1, phase_train=phase_train)
    print('End Inception', conv)

    # Linear layers
    fc = sdn.fc7_layer('FC', conv, 16, True, phase_train, FLAGS.dropout_factor, BN=True, override=3)
    fc = sdn.linear_layer('Linear', fc, 8, False, phase_train, BN=True)
    Logits = sdn.linear_layer('Output', fc, FLAGS.num_classes, False, phase_train, BN=False, relu=False, add_bias=False)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma), conv


def forward_pass_dense(images, phase_train):

    """
    Perform forward pass using a DenseNEt
    :param images: 3 channel images
    :param phase_train: training or testing phase
    :return: logits, l2 loss and last conv layaer
    """

    # Define densenet class
    dense = SDN.DenseNet(nb_blocks=5, filters=6, sess=None, phase_train=phase_train, summary=False)
    print('Input Images: ', images)

    # First layer is conv
    conv = sdn.convolution('Conv1', images, 3, 16, 2, phase_train=phase_train)

    # 5 Dense blocks
    conv = dense.dense_block(conv, nb_layers=4, layer_name='Dense64', downsample=True)
    conv = dense.dense_block(conv, nb_layers=8, layer_name='Dense32', downsample=True)
    conv = dense.dense_block(conv, nb_layers=16, layer_name='Dense16', downsample=True)
    conv = dense.dense_block(conv, nb_layers=24, layer_name='Dense8', downsample=True)
    conv = dense.dense_block(conv, nb_layers=48, layer_name='Dense4', downsample=False, keep_prob=FLAGS.dropout_factor)
    print ('End Dense: ', conv) # , keep_prob=FLAGS.dropout_factor

    # Linear layers
    fc = sdn.fc7_layer('FC', conv, 16, True, phase_train, FLAGS.dropout_factor, BN=True, override=3)
    fc = sdn.linear_layer('Linear', fc, 8, False, phase_train, BN=True)
    Logits = sdn.linear_layer('Output', fc, FLAGS.num_classes, False, phase_train, BN=False, relu=False, add_bias=False)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma), conv


def total_loss(logits, labels):

    """
    Add loss to the trainable variables and a summary
    :param logits:
    :param labels:
    :return:
    """

    if FLAGS.loss_factor != 1.0:

        # Make a nodule sensitive binary for specified values
        lesion_mask = tf.cast(labels == 0, tf.float32)

        # Now multiply this mask by scaling factor then add back to labels. Add 1 to prevent 0 loss
        lesion_mask = tf.add(tf.multiply(lesion_mask, FLAGS.loss_factor), 1)

    # Change labels to one hot
    labels = tf.one_hot(tf.cast(labels, tf.uint8), depth=FLAGS.num_classes, dtype=tf.uint8)

    # Calculate  loss
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.squeeze(labels), logits=logits)

    # Multiply the loss factor
    if FLAGS.loss_factor != 1.0: loss = tf.multiply(loss, tf.squeeze(lesion_mask))

    # Reduce to scalar
    loss = tf.reduce_mean(loss)

    # Output the losses
    # Acc = tf.contrib.metrics.accuracy(tf.argmax(logits, 1), tf.cast(labels, tf.int64))
    # tf.summary.scalar('Accuracy', Acc[1])
    tf.summary.scalar('Cross Entropy', loss)

    if FLAGS.num_classes == 1:
        AUC = tf.contrib.metrics.streaming_auc(tf.argmax(logits, 1), labels)
        tf.summary.scalar('AUC', AUC[1])

    # Add these losses to the collection
    tf.add_to_collection('losses', loss)

    return loss


def backward_pass(total_loss):

    """
    This function performs our backward pass and updates our gradients
    :param total_loss:
    :return:
    """

    # Get the tensor that keeps track of step in this graph or create one if not there
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', total_loss)

    # Compute the gradients. NAdam optimizer came in tensorflow 1.2
    opt = tf.contrib.opt.NadamOptimizer(learning_rate=FLAGS.learning_rate, beta1=FLAGS.beta1,
                                        beta2=FLAGS.beta2, epsilon=1e-8)

    # Compute the gradients
    gradients = opt.compute_gradients(total_loss)

    # Apply the gradients
    train_op = opt.apply_gradients(gradients, global_step, name='train')

    # Add histograms for the trainable variables. i.e. the collection of variables created with Trainable=True
    for var in tf.trainable_variables(): tf.summary.histogram(var.op.name, var)

    # Maintain average weights to smooth out training
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_avg_decay, global_step)

    # Applies the average to the variables in the trainable ops collection
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # Does nothing. placeholder to control the execution of the graph
    with tf.control_dependencies([train_op, variable_averages_op]): dummy_op = tf.no_op(name='train')

    return dummy_op


def inputs(skip=False):

    """
    This function loads our raw inputs, processes them to a protobuffer that is then saved and
    loads the protobuffer into a batch of tensors
    """

    # To Do: Skip part 1 and 2 if the protobuff already exists
    if not skip: Input.pre_process(FLAGS.box_dims, 128)

    print('----------------------------------------Loading Protobuff...')
    train = Input.load_protobuf()
    valid = Input.load_validation_set()


    return train, valid