"""
Does our loading and preprocessing of files to a protobuf
"""

import glob

import numpy as np
import tensorflow as tf
import SODLoader as SDL
from pathlib import Path

from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/BreastData/Mammo/RiskStudy'
brca_dir = str(Path.home()) + '/PycharmProjects/Datasets/BreastData/Mammo/BRCA'

sdl = SDL.SODLoader(data_root=home_dir)


def pre_process(box_dims=512):

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

        # Augment the low class
        if label == 1: copies = 5
        else: copies = 1

        for _ in range (copies):

            # Save an example
            data[index] = {'data': image.astype(np.float32), 'label_data': label_data.astype(np.float32), 'file': file, 'shapex': shape[0],
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


def pre_process_BRCA(box_dims=512):

    """
    Loads the files to a protobuf
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, brca_dir)
    shuffle(filenames)
    print (len(filenames), 'Base Files: ', filenames)

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0], {}, {}, 0, 0

    for file in filenames:

        # Retreive patient number
        group = file.split('/')[-4]
        patient = group + file.split('/')[-3].split(' ')[-1]
        view = file.split('/')[-2]

        if 'Neg' in group: cancer = 0
        else: cancer = 1

        # These are all BRCA so all high risk
        label = 1

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

        # Augment the low class
        if label == 1: copies = 5
        else: copies = 1

        # Save the first 100 as testing
        if index < 100: data_test[index] = {'data': image.astype(np.float32), 'label_data': label_data.astype(np.float32), 'file': file, 'shapex': shape[0],
                           'shapy': shape[1], 'group': group, 'patient': patient, 'class_raw': view, 'label': label, 'accno': accno}

        else: data[index-100] = {'data': image.astype(np.float32), 'label_data': label_data.astype(np.float32), 'file': file, 'shapex': shape[0],
                           'shapy': shape[1], 'group': group, 'patient': patient, 'class_raw': view, 'label': label, 'accno': accno}

        # Increment counter
        index += 1
        counter[label] += 1

        # Done with this patient
        pt += 1

    # # Done with all patients
    print ('Made %s boxes from %s patients. Train: %s Test: %s' %(index, pt, len(data), len(data_test)))

    # Save the data
    sdl.save_tfrecords(data, 1, file_root='data/BRCA_Train_')
    sdl.save_tfrecords(data_test, 1, file_root='data/BRCA_Test_')


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
        if FLAGS.test_files not in filenames1[i]: filenames.append(filenames1[i])

    # Show the file names
    print('Training files: %s' % filenames)

    # Load the dictionary
    data = sdl.load_tfrecords(filenames, FLAGS.box_dims, tf.float32, channels=1)

    # Data Augmentation ------------------

    # Random rotate
    angle = tf.random_uniform([], -0.45, 0.45)
    data['data'], data['label_data'] = tf.contrib.image.rotate(data['data'], angle), tf.contrib.image.rotate(data['label_data'], angle)

    # Random crop
    resize_factor = int(FLAGS.network_dims/0.9)
    data['data'] = tf.image.resize_images(data['data'], [resize_factor, resize_factor])
    data['label_data'] = tf.image.resize_images(data['label_data'], [resize_factor, resize_factor])
    data['data'] = tf.random_crop(data['data'], [FLAGS.network_dims, FLAGS.network_dims, 1])
    data['label_data'] = tf.random_crop(data['label_data'], [FLAGS.network_dims, FLAGS.network_dims, 1])

    # Random shear:
    rand = []
    for z in range(4):
        rand.append(tf.random_uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32))
    data['data'] = tf.contrib.image.transform(data['data'], [1, rand[0], rand[1], rand[2], 1, rand[3], 0, 0])
    data['label_data'] = tf.contrib.image.transform(data['label_data'], [1, rand[0], rand[1], rand[2], 1, rand[3], 0, 0])

    # Randomly flip
    def flip(mode=None):

        if mode == 1: img, lbl = tf.image.flip_up_down(data['data']), tf.image.flip_up_down(data['label_data'])
        elif mode == 2: img, lbl = tf.image.flip_left_right(data['data']), tf.image.flip_left_right(data['label_data'])
        else: img, lbl = data['data'], data['label_data']
        return img, lbl

    data['data'], data['label_data'] = tf.cond(tf.squeeze(tf.random_uniform([1], 0, 2, dtype=tf.int32)) > 0, lambda: flip(1), lambda: flip(0))
    data['data'], data['label_data'] = tf.cond(tf.squeeze(tf.random_uniform([1], 0, 2, dtype=tf.int32)) > 0, lambda: flip(2), lambda: flip(0))

    # # Random contrast and brightness
    # data['data'] = tf.image.random_brightness(data['data'], max_delta=2)
    # data['data'] = tf.image.random_contrast(data['data'], lower=0.975, upper=1.025)
    #
    # # Random gaussian noise
    # T_noise = tf.random_uniform([], 0, 0.1)
    # noise = tf.random_uniform(shape=[FLAGS.network_dims, FLAGS.network_dims, 1], minval=-T_noise, maxval=T_noise)
    # data['data'] = tf.add(data['data'], tf.cast(noise, tf.float32))

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
    filenames = []

    # Retreive only the right filename
    for i in range(0, len(filenames1)):
        if FLAGS.test_files in filenames1[i]: filenames.append(filenames1[i])

    print('Testing files: %s' % filenames)

    # Load the dictionary
    data = sdl.load_tfrecords(filenames, FLAGS.box_dims, tf.float32, channels=1)

    # Reshape image
    data['data'] = tf.image.resize_images(data['data'], [FLAGS.network_dims, FLAGS.network_dims])
    data['label_data'] = tf.image.resize_images(data['label_data'], [FLAGS.network_dims, FLAGS.network_dims])

    # Display the images
    tf.summary.image('Test IMG', tf.reshape(data['data'], shape=[1, FLAGS.network_dims, FLAGS.network_dims, 1]), 4)

    return sdl.val_batches(data, FLAGS.batch_size)