"""
Does our loading and preprocessing of files to a protobuf
"""

import os, glob, cv2

import numpy as np
import tensorflow as tf
import SODLoader as SDL
from pathlib import Path

from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/BreastData/Mammo/RiskStudy'

sdl = SDL.SODLoader(data_root=home_dir)


def pre_process(warps, box_dims=512):

    """
    Loads the files to a protobuf
    :param warps:
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, home_dir)
    shuffle(filenames)
    print (len(filenames), 'Base Files: ', filenames)

    # Global variables
    display, counter, data, index, pt = [], [0, 0], {}, 0, 0

    for file in filenames:

        # Retreive patient number
        group = file.split('/')[-4]
        patient = file.split('/')[-2] + group
        class_raw = file.split('/')[-3]
        if 'Normal' in class_raw: label = 0
        else: label = 1

        # Load and resize image
        try: image, accno, shape, _, _ = sdl.load_DICOM_2D(file)
        except:
            print ("Failed to load: ", file)
            continue

        image = sdl.zoom_2D(image, [box_dims, box_dims])

        # Create and apply mask/labels
        mask = sdl.create_mammo_mask(image)
        label_data = np.multiply(mask.astype(np.uint8), (label+1))

        # Normalize image, mean/std: 835.3 1189.5
        image = (image - 835.3) / 1189.5
        image *= mask

        # Save an example
        data[index] = {'data': image.astype(np.float32), 'label_data': label_data.astype(np.uint8), 'file': file, 'shapex': shape[0],
                       'shapy': shape[1], 'group': group, 'patient': patient, 'class_raw': class_raw, 'label': label, 'accno': accno}

        # Increment counter
        index += 1
        counter[label] += 1

        # Done with this patient
        pt += 1

    # # Done with all patients
    print ('Made %s boxes from %s patients. Class counts: %s' %(index, pt, counter))

    # Save the data
    sdl.save_tfrecords(data, 4, file_root='data/Breast_')
    sdl.save_dict_filetypes(data[0])


def load_protobuf():

    """
    Loads the protocol buffer into a form to send to shuffle
    :param 
    :return:
    """

    # Load all the filenames in glob
    filenames1 = glob.glob('data/*.tfrecords')
    filenames = []

    # Define the filenames to remove
    for i in range(0, len(filenames1)):
        if FLAGS.test_files not in filenames1[i]:
            filenames.append(filenames1[i])

    # Show the file names
    print('Training files: %s' % filenames)

    # Load the dictionary
    data = sdl.load_tfrecords(filenames, FLAGS.box_dims, tf.float32, channels=1)

    # Image augmentation
    angle = tf.random_uniform([1], -1.78, 1.571)

    # First randomly rotate
    data['data'] = tf.contrib.image.rotate(data['data'], angle)

    # Then randomly flip
    data['data'] = tf.image.random_flip_left_right(tf.image.random_flip_up_down(data['data']))

    # Color image randomization
    tf.summary.image('Pre-Contrast', tf.reshape(data['data'], shape=[1, FLAGS.box_dims, FLAGS.box_dims, 1]), 4)
    data['data'] = tf.image.random_contrast(data['data'], lower=0.9, upper=1.1)

    # Reshape image
    data['data'] = tf.image.resize_images(data['data'], [FLAGS.network_dims, FLAGS.network_dims])

    # Display the images
    tf.summary.image('Train IMG', tf.reshape(data['data'], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 4)

    # Return data dictionary
    return sdl.randomize_batches(data, FLAGS.batch_size)


def load_validation_set():

    """
        Same as load protobuf() but loads the validation set
        :return:
    """

    # Use Glob here
    filenames1 = glob.glob('data/*.tfrecords')

    # The real filenames
    filenames = []

    # Retreive only the right filename
    for i in range(0, len(filenames1)):
        if FLAGS.test_files in filenames1[i]:
            filenames.append(filenames1[i])

    print('Testing files: %s' % filenames)

    # Load the dictionary
    data = sdl.load_tfrecords(filenames, FLAGS.box_dims, tf.float32, channels=1)

    # Reshape image
    data['data'] = tf.image.resize_images(data['data'], [FLAGS.network_dims, FLAGS.network_dims])

    # Display the images
    tf.summary.image('Test IMG', tf.reshape(data['data'], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 4)

    return sdl.val_batches(data, FLAGS.batch_size)

pre_process(1)