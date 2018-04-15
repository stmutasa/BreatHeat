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


def forward_pass_fancy(images, phase_train):

    """
    This function builds the network architecture and performs the forward pass
    Two main architectures depending on where to insert the inception or residual layer
    :param images: Images to analyze
    :param phase_train1: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """

    print ('Input images: ', images)

    # Network blocks
    conv1 = sdn.convolution('Conv1', images, 3, 8, 1, phase_train=phase_train)
    conv1 = sdn.convolution('Conv1b', conv1, 3, 8, 1, phase_train=phase_train)
    down = sdn.convolution('Down128', conv1, 2, 16, 2, phase_train=phase_train)
    print('*' * 30, conv1, down)

    conv2 = sdn.convolution('Conv2', down, 3, 16, 1, phase_train=phase_train)
    conv2 = sdn.residual_layer('Conv2b', conv2, 3, 16, 1, phase_train=phase_train)
    down = sdn.convolution('Down64', conv2, 2, 32, 2, phase_train=phase_train)
    print('*' * 22, conv2)

    conv3 = sdn.residual_layer('Conv3', down, 3, 32, 1, phase_train=phase_train)
    conv3 = sdn.residual_layer('Conv3b', conv3, 3, 32, 1, phase_train=phase_train)
    down = sdn.convolution('Down32', conv3, 2, 64, 2, phase_train=phase_train) # Now 32x32
    print('*'*14,conv3)

    conv4 = sdn.residual_layer('Conv4', down, 3, 64, 1, phase_train=phase_train)
    conv4 = sdn.residual_layer('Conv4b', conv4, 3, 64, 1, phase_train=phase_train)
    down = sdn.convolution('Down16', conv4, 2, 128, 2, phase_train=phase_train)
    print('*'*6,conv4)

    conv5 = sdn.inception_layer('Conv5', down, 128, 1, phase_train=phase_train)
    conv5 = sdn.inception_layer('Conv5b', conv5, 128, 1, phase_train=phase_train)
    down = sdn.convolution('Down8', conv5, 2, 256, 2, phase_train=phase_train)
    print('*' * 3, conv5)

    conv6 = sdn.inception_layer('Conv6', down, 256, phase_train=phase_train)
    conv6 = sdn.inception_layer('Conv6b', conv6, 256, phase_train=phase_train)
    down = sdn.convolution('Down4', conv6, 2, 512, 2, phase_train=phase_train)
    print(conv6)

    # Bottom of the decoder: 4x4
    conv7 = sdn.inception_layer('Bottom1', down, 512, phase_train=phase_train)
    conv7 = sdn.residual_layer('Bottom2', conv7, 3, 512, 1, dropout=FLAGS.dropout_factor, phase_train=phase_train)
    conv7 = sdn.inception_layer('Bottom2', conv7, 512, phase_train=phase_train)

    # Upsample 1
    dconv = sdn.deconvolution('Dconv1', conv7, 2, 256, S=2, phase_train=phase_train, concat=False, concat_var=conv6, out_shape=[FLAGS.batch_size, 8, 8, 256])
    dconv = sdn.inception_layer('Dconv1b', dconv, 256, phase_train=phase_train)
    print('-'*3, dconv)

    dconv = sdn.deconvolution('Dconv2', dconv, 2, 128, S=2, phase_train=phase_train, concat=False, concat_var=conv5, out_shape=[FLAGS.batch_size, 16, 16, 128])
    dconv = sdn.inception_layer('Dconv2b', dconv, 128, phase_train=phase_train)
    print('-' * 6, dconv)

    dconv = sdn.deconvolution('Dconv3', dconv, 2, 64, S=2, phase_train=phase_train, concat=False, concat_var=conv4, out_shape=[FLAGS.batch_size, 32, 32, 64])
    dconv = sdn.inception_layer('Dconv3b', dconv, 64, phase_train=phase_train)
    print ('-'*14, dconv)

    dconv = sdn.deconvolution('Dconv4', dconv, 2, 32, S=2, phase_train=phase_train, concat=False, concat_var=conv3, out_shape=[FLAGS.batch_size, 64, 64, 32])
    dconv = sdn.residual_layer('Dconv4b', dconv, 3, 32, S=1, phase_train=phase_train)
    print ('-'*22, dconv)

    dconv = sdn.deconvolution('Dconv5', dconv, 2, 16, S=2, phase_train=phase_train, concat=False, concat_var=conv2, out_shape=[FLAGS.batch_size, 128, 128, 16])
    dconv = sdn.residual_layer('Dconv5b', dconv, 3, 16, S=1, phase_train=phase_train)
    print ('-'*30, dconv)

    dconv = sdn.deconvolution('Dconv6', dconv, 2, 8, S=2, phase_train=phase_train, concat=False, concat_var=conv1, out_shape=[FLAGS.batch_size, 256, 256, 8])
    dconv = sdn.convolution('Dconv6b', dconv, 3, 8, S=1, phase_train=phase_train, dropout=FLAGS.dropout_factor)

    # Output is a 1x1 box with 3 labels
    Logits = sdn.convolution('Logits', dconv, 1, FLAGS.num_classes, S=1, phase_train=phase_train, BN=False, relu=False, bias=False)
    print ('Logits: ', Logits)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma)


def forward_pass_class(images, phase_train):

    """
    This function builds the network architecture and performs the forward pass
    Two main architectures depending on where to insert the inception or residual layer
    :param images: Images to analyze
    :param phase_train1: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """

    K = 16

    # First layer is conv
    print('Input Images: ', images)

    # Residual blocks
    conv = sdn.convolution('Conv1', images, 3, K, 2, phase_train=phase_train) # 128
    conv = sdn.residual_layer('Residual1', conv, 3, K * 2, 2, phase_train=phase_train)
    conv = sdn.residual_layer('Residual2', conv, 3, K * 4, 2, phase_train=phase_train)
    conv = sdn.residual_layer('Residual3', conv, 3, K * 8, 2, phase_train=phase_train)  # 16x16
    conv = sdn.residual_layer('Residual4', conv, 3, K * 8, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Residual5', conv, 3, K * 16, 2, phase_train=phase_train)
    conv = sdn.inception_layer('Inception6', conv, K * 16, S=1, phase_train=phase_train)
    conv = sdn.residual_layer('Residual7', conv, 3, K * 32, 2, phase_train=phase_train)
    conv = sdn.residual_layer('Residual8', conv, 3, K * 32, 1, phase_train=phase_train)
    conv = sdn.residual_layer('Residual9', conv, 3, K * 32, 1, phase_train=phase_train)

    # # Define densenet class
    # dense = SDN.DenseNet(nb_blocks=6, filters=6, sess=None, phase_train=phase_train, summary=False)
    # conv = sdn.convolution('Conv1', images, 3, 16, 2, phase_train=phase_train)
    # conv = dense.dense_block(conv, nb_layers=2, layer_name='Dense128', downsample=True)
    # conv = dense.dense_block(conv, nb_layers=4, layer_name='Dense64', downsample=True)
    # conv = dense.dense_block(conv, nb_layers=8, layer_name='Dense32', downsample=True)
    # conv = dense.dense_block(conv, nb_layers=16, layer_name='Dense16', downsample=True)
    # conv = dense.dense_block(conv, nb_layers=24, layer_name='Dense8', downsample=True)
    # conv = dense.dense_block(conv, nb_layers=48, layer_name='Dense4', downsample=False, keep_prob=FLAGS.dropout_factor)

    print('End Dims', conv)

    # Linear layers
    fc = sdn.fc7_layer('FC', conv, 16, True, phase_train, FLAGS.dropout_factor, BN=True)
    fc = sdn.linear_layer('Linear', fc, 8, False, phase_train, BN=True)
    Logits = sdn.linear_layer('Output', fc, FLAGS.num_classes, False, phase_train, BN=False, relu=False, add_bias=False)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma)


def forward_pass_extend(images, phase_train):

    """
    This function builds the network architecture and performs the forward pass
    Two main architectures depending on where to insert the inception or residual layer
    :param images: Images to analyze
    :param phase_train1: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """

    print ('Input images: ', images)

    # Network blocks
    conv1 = sdn.convolution('Conv1', images, 3, 8, 1, phase_train=phase_train)
    conv1 = sdn.convolution('Conv1b', conv1, 3, 8, 1, phase_train=phase_train)
    down = sdn.convolution('Down128', conv1, 2, 16, 2, phase_train=phase_train)
    print('*' * 30, conv1, down)

    conv2 = sdn.convolution('Conv2', down, 3, 16, 1, phase_train=phase_train)
    conv2 = sdn.residual_layer('Conv2b', conv2, 3, 16, 1, phase_train=phase_train)
    down = sdn.convolution('Down64', conv2, 2, 32, 2, phase_train=phase_train)
    print('*' * 22, conv2)

    conv3 = sdn.residual_layer('Conv3', down, 3, 32, 1, phase_train=phase_train)
    conv3 = sdn.residual_layer('Conv3b', conv3, 3, 32, 1, phase_train=phase_train)
    down = sdn.convolution('Down32', conv3, 2, 64, 2, phase_train=phase_train) # Now 32x32
    print('*'*14,conv3)

    conv4 = sdn.residual_layer('Conv4', down, 3, 64, 1, phase_train=phase_train)
    conv4 = sdn.residual_layer('Conv4b', conv4, 3, 64, 1, phase_train=phase_train)
    down = sdn.convolution('Down16', conv4, 2, 128, 2, phase_train=phase_train)
    print('*'*6,conv4)

    conv5 = sdn.inception_layer('Conv5', down, 128, 1, phase_train=phase_train)
    conv5 = sdn.inception_layer('Conv5b', conv5, 128, 1, phase_train=phase_train)
    down = sdn.convolution('Down8', conv5, 2, 256, 2, phase_train=phase_train)
    print('*' * 3, conv5)

    conv6 = sdn.inception_layer('Conv6', down, 256, phase_train=phase_train)
    conv6 = sdn.inception_layer('Conv6b', conv6, 256, phase_train=phase_train)
    down = sdn.convolution('Down4', conv6, 2, 512, 2, phase_train=phase_train)
    print(conv6)

    # Bottom of the decoder: 4x4
    conv7 = sdn.inception_layer('Bottom1', down, 512, phase_train=phase_train)
    conv7 = sdn.residual_layer('Bottom2', conv7, 3, 512, 1, dropout=FLAGS.dropout_factor, phase_train=phase_train)
    conv7 = sdn.inception_layer('Bottom2', conv7, 512, phase_train=phase_train)

    # Upsample 1
    dconv = sdn.deconvolution('Dconv1', conv7, 2, 256, S=2, phase_train=phase_train, concat=False, concat_var=conv6, out_shape=[FLAGS.batch_size, 8, 8, 256])
    dconv = sdn.inception_layer('Dconv1b', dconv, 256, phase_train=phase_train)
    print('-'*3, dconv)

    dconv = sdn.deconvolution('Dconv2', dconv, 2, 128, S=2, phase_train=phase_train, concat=False, concat_var=conv5, out_shape=[FLAGS.batch_size, 16, 16, 128])
    dconv = sdn.inception_layer('Dconv2b', dconv, 128, phase_train=phase_train)
    print('-' * 6, dconv)

    dconv = sdn.deconvolution('Dconv3', dconv, 2, 64, S=2, phase_train=phase_train, concat=False, concat_var=conv4, out_shape=[FLAGS.batch_size, 32, 32, 64])
    dconv = sdn.inception_layer('Dconv3b', dconv, 64, phase_train=phase_train)
    print ('-'*14, dconv)

    dconv = sdn.deconvolution('Dconv4', dconv, 2, 32, S=2, phase_train=phase_train, concat=False, concat_var=conv3, out_shape=[FLAGS.batch_size, 64, 64, 32])
    dconv = sdn.residual_layer('Dconv4b', dconv, 3, 32, S=1, phase_train=phase_train)
    print ('-'*22, dconv)

    dconv = sdn.deconvolution('Dconv5', dconv, 2, 16, S=2, phase_train=phase_train, concat=False, concat_var=conv2, out_shape=[FLAGS.batch_size, 128, 128, 16])
    dconv = sdn.residual_layer('Dconv5b', dconv, 3, 16, S=1, phase_train=phase_train)
    print ('-'*30, dconv)

    dconv = sdn.deconvolution('Dconv6', dconv, 2, 8, S=2, phase_train=phase_train, concat=False, concat_var=conv1, out_shape=[FLAGS.batch_size, 256, 256, 8])
    dconv = sdn.residual_layer('Dconv6a', dconv, 3, 8, S=1, phase_train=phase_train)
    dconv = sdn.residual_layer('Dconv6b', dconv, 3, 8, S=1, phase_train=phase_train)
    dconv = sdn.residual_layer('Dconv6c', dconv, 3, 8, S=1, phase_train=phase_train, dropout=FLAGS.dropout_factor)

    # Output is a 1x1 box with 3 labels
    Logits = sdn.convolution('Logits', dconv, 1, FLAGS.num_classes, S=1, phase_train=phase_train, BN=False, relu=False, bias=False)
    print ('Logits: ', Logits)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma)


def forward_pass(images, phase_train):

    """
    This function builds the network architecture and performs the forward pass
    Two main architectures depending on where to insert the inception or residual layer
    :param images: Images to analyze
    :param phase_train1: bool, whether this is the training phase or testing phase
    :return: logits: the predicted age from the network
    :return: l2: the value of the l2 loss
    """

    print ('Input images: ', images)

    # Network blocks
    conv1 = sdn.convolution('Conv1', images, 3, 16, 1, phase_train=phase_train)
    down = sdn.convolution('Down128', conv1, 3, 32, 2, phase_train=phase_train)
    print('*' * 30, conv1, down)

    conv2 = sdn.convolution('Conv2', down, 3, 32, 1, phase_train=phase_train)
    down = sdn.convolution('Down64', conv2, 3, 64, 2, phase_train=phase_train)
    print('*' * 22, conv2)

    conv3 = sdn.convolution('Conv3', down, 3, 64, 1, phase_train=phase_train)
    down = sdn.convolution('Down32', conv3, 3, 128, 2, phase_train=phase_train) # Now 32x32
    print('*'*14,conv3)

    conv4 = sdn.convolution('Conv4', down, 3, 128, 1, phase_train=phase_train)
    down = sdn.convolution('Down16', conv4, 3, 256, 2, phase_train=phase_train)
    print('*'*6,conv4)


    # Bottom of the decoder: 16x16
    conv5 = sdn.convolution('Bottom1', down, 3, 256, 1, phase_train=phase_train)
    conv5 = sdn.convolution('Bottom2', conv5, 3, 256, 1, phase_train=phase_train)
    conv5 = sdn.convolution('Bottom3', conv5, 3, 256, 1, phase_train=phase_train)

    # Upsamples
    dconv = sdn.deconvolution('Dconv2', conv5, 2, 128, S=2, phase_train=phase_train, concat=False, concat_var=conv4, out_shape=[FLAGS.batch_size, 32, 32, 128])
    dconv = sdn.convolution('Dconv2b', dconv, 3, 128, S=1, phase_train=phase_train)
    print('-' * 6, dconv)

    dconv = sdn.deconvolution('Dconv3', dconv, 2, 64, S=2, phase_train=phase_train, concat=False, concat_var=conv3, out_shape=[FLAGS.batch_size, 64, 64, 64])
    dconv = sdn.convolution('Dconv3b', dconv, 3, 64, S=1, phase_train=phase_train)
    print ('-'*14, dconv)

    dconv = sdn.deconvolution('Dconv4', dconv, 2, 32, S=2, phase_train=phase_train, concat=False, concat_var=conv2, out_shape=[FLAGS.batch_size, 128, 128, 32])
    dconv = sdn.convolution('Dconv4b', dconv, 3, 32, S=1, phase_train=phase_train)
    print ('-'*22, dconv)

    dconv = sdn.deconvolution('Dconv5', dconv, 2, 16, S=2, phase_train=phase_train, concat=False, concat_var=conv1, out_shape=[FLAGS.batch_size, 256, 256, 16])
    dconv = sdn.convolution('Dconv5b', dconv, 3, 16, S=1, phase_train=phase_train)
    print ('-'*30, dconv)

    # Output is a 1x1 box with 3 labels
    Logits = sdn.convolution('Logits', dconv, 1, FLAGS.num_classes, S=1, phase_train=phase_train, BN=False, relu=False, bias=False, dropout=FLAGS.dropout_factor)
    print ('Logits: ', Logits)

    return Logits, sdn.calc_L2_Loss(FLAGS.l2_gamma)


def total_loss(logitz, labelz, num_classes=2, loss_type=None):

    """
    Cost function
    :param logitz: The raw log odds units output from the network
    :param labelz: The labels: not one hot encoded
    :param num_classes: number of classes predicted
    :param class_weights: class weight array
    :param loss_type: DICE or other to use dice or weighted
    :return:
    """

    # Reduce dimensionality
    labels, logits = tf.cast(tf.squeeze(labelz), tf.uint8), tf.squeeze(logitz)

    # Summary images
    im_num = int(FLAGS.batch_size / 2)
    tf.summary.image('Labels', tf.reshape(tf.cast(labelz[im_num], tf.float32), shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)
    tf.summary.image('Logits', tf.reshape(logitz[im_num, : , : , 1], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 2)

    if loss_type=='DICE':

        # Remove background label
        # labels = tf.cast(labelz > 1, tf.uint8)

        # Make labels one hot
        labels = tf.cast(tf.one_hot(labels, depth=FLAGS.num_classes, dtype=tf.uint8), tf.float32)

        # Generate and apply mask
        # mask = tf.expand_dims(tf.cast(labelz > 0, tf.float32), -1)
        # logits, labels = logitz * mask, labels * mask

        # Flatten
        logits = tf.reshape(logits, [-1, num_classes])
        labels = tf.reshape(labels, [-1, num_classes])

        # To prevent number errors:
        eps = 1e-5

        # Calculate softmax:
        logits = tf.nn.softmax(logits)

        # Find the intersection
        intersection = 2*tf.reduce_sum(logits * labels)

        # find the union
        union = eps + tf.reduce_sum(logits) + tf.reduce_sum(labels)

        # Calculate the loss
        dice = intersection/union

        # Output the training DICE score
        tf.summary.scalar('DICE_Score', dice)

        # 1-DICE since we want better scores to have lower loss
        loss = 1 - dice

    else:

        # Remove background label
        # labels = tf.cast(labelz > 1, tf.uint8)

        # Make labels one hot
        labels = tf.cast(tf.one_hot(labels, depth=FLAGS.num_classes, dtype=tf.uint8), tf.float32)

        # # Generate and apply mask
        # mask = tf.expand_dims(tf.cast(labelz > 0, tf.float32), -1)
        # logits, labels = logitz * mask, labels * mask

        # Flatten
        logits = tf.reshape(logitz, [-1, num_classes])
        labels = tf.cast(tf.reshape(labels, [-1, num_classes]), tf.float32)

        # Calculate the loss: Result is batch x 65k
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

        # Reduce the loss into a scalar
        loss = tf.reduce_mean(loss)

    # Output the Loss
    tf.summary.scalar('Loss_Raw', loss)

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
    global_step = tf.train.get_or_create_global_step()

    # Print summary of total loss
    tf.summary.scalar('Total_Loss', total_loss)

    # Compute the gradients..2
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

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
    if not skip:
        Input.pre_process(FLAGS.box_dims)
        Input.pre_process_BRCA(FLAGS.box_dims)

    print('----------------------------------------Loading Protobuff...')
    train = Input.load_protobuf()
    valid = Input.load_validation_set()


    return train, valid