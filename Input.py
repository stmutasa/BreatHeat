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

def create_mammo_mask(image, threshold=2, size_denominator=45):

    """

    :param image:
    :param threshold:
    :param size_denominator:
    :return:
    """

    # Create the mask
    mask = np.copy(image)

    # Apply gaussian blur to smooth the image
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    sdl.display_single_image(mask, title='Blur')
    # mask = cv2.bilateralFilter(mask.astype(np.float32),9,75,75)

    # Threshold the image
    mask = np.squeeze(mask > threshold)
    sdl.display_single_image(mask, title='Threshold')

    # Define the CV2 structuring element
    radius_close = np.round(mask.shape[1] / size_denominator).astype('int16')
    kernel_close = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(radius_close, radius_close))

    # Apply morph close
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    sdl.display_single_image(mask, title='close')

    # Invert mask
    mask = ~mask
    sdl.display_single_image(mask, title='Inverted')

    # Add 2
    mask += 2
    sdl.display_single_image(mask, title='Plus2')

    return mask

def pre_process(warps):

    """
    Loads the files to a protobuf
    :param warps:
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, home_dir)
    shuffle(filenames)
    print (len(filenames), 'Base Files: ', filenames)

    # Global variables
    display, writer, counter, data, index, pt = [], [], [None, None], {}, 0, 0

    for file in filenames:

        # Retreive patient number
        group = file.split('/')[-4]
        patient = file.split('/')[-2] + group
        class_raw = file.split('/')[-3]
        if 'Normal' in class_raw: label = 0
        else: label = 1

        # load image
        image, accno, dims, _, _ = sdl.load_DICOM_2D(file)

        sdl.display_single_image(image, title='Image')
        create_mammo_mask(image)

        # TODO test
        print (image.shape, accno, dims)
        pt +=1
        if pt>5: break

    #     # Create background mask
    #
    #     # Save an example
    #     data[index] = {'data': box, 'patient': patient, 'file': file, 'age': age, 'dx': dx, 'label': label,
    #                    'label2': label2, 'sex': sex, 'race': race, 'race_name': race_name, 'case': case, 'nerve': nerve}
    #
    #     # Increment counter
    #     index += 1
    #
    #     # Done with this patient
    #     pt += 1
    #
    # # # Done with all patients
    # print ('Made %s boxes from %s patients. Class counts: %s' %(index, pt, counter))
    #
    # # Save the data
    # sdl.save_tfrecords(data, 1, file_root='data/Breast_')
    # sdl.save_dict_filetypes(data[0])


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