"""
Does our loading and preprocessing of files to a protobuf

Who are the patients who have cancer in our dataset: (1885 / 1824  no cancer views, 356 / 356 (+407 Ca+) cancer views)

BRCA:
	(37*4 or 146) BRCA/G1_PosBRCA/Cancer/Patient 2/R CC/xxx.dcm (4 views each patient)
	# /BRCA/G3_Neg_NoMutations/3/Serxx.dcm
    # /BRCA/G3_Neg_MinorMutations/3/Serxx.dcm
    # /BRCA/G1_PosBRCA/NoCancer/Patient 1/L CC/Ser.dcm
    # /BRCA/G1_PosBRCA/Cancer/Patient 1/L CC/Ser.dcm
    # TODO: Make high risk if positive for BRCA

Calcs:
    File 1 and File 2 contain CC and MLO views of the affected breast - These pts have cancer though
    The /new data does not contain full field mammograms - dammit man!
	Invasive
	Microinvasion
	(ADH and DCIS is not cancer)
	# TODO: All high risk

Chemoprevention:
	(All just high risk ADH, LCIS and DCIS)
	# TODO: All high risk

RiskStudy:
	(201*2) RiskStudy/HighRisk/Cancer/4/imgxx.dcm (2 views each pt.)
	(Low risk is low risk with no cancer)
	(High Risk / Normal didnt get cancer)
	TODO: Make high risk if they got cancer (unclear how high risk grp determined)
"""

import subprocess

# # Download the SODKit
# args = ['git', 'clone', 'https://github.com/stmutasa/SODKit']
# subprocess.run(args)

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD
from pathlib import Path

from random import shuffle
import os

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = '/media/stmutasa/0B4DFD6B47E23585/Datasets/BreastData/Mammo/'
risk_dir = home_dir + 'RiskStudy/'
brca_dir = home_dir + 'BRCA/'
calc_dir = home_dir + 'Calcs/Eduardo/'
chemo_dir = home_dir + 'Chemoprevention/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()

def pre_process_BRCA(box_dims=1024):

    """
    Loads the files to a protobuf
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, brca_dir)
    shuffle(filenames)

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0], {}, {}, 0, 0

    for file in filenames:

        """
        Retreive patient number, two different if statements because...
        
        # /BRCA/G3_Neg_NoMutations/3/Serxx.dcm
        # /BRCA/G3_Neg_MinorMutations/3/Serxx.dcm
        # /BRCA/G1_PosBRCA/NoCancer/Patient 1/L CC/Ser.dcm
        # /BRCA/G1_PosBRCA/Cancer/Patient 1/L CC/Ser.dcm
        
        We want: group = source of positive, brca vs risk 
        Patient = similar to accession, (can have multiple views) (BRCApos_Cancer_1, BRCAneg_NoMutations_1)
        View = unique to that view (BRCA_Cancer_1_LCC)
        Label = 1 if cancer, 0 if not 
        """

        group = 'BRCA'

        if 'G1_PosBRCA' in file:
            patient = 'BRCApos_' + file.split('/')[-4] + '_' + file.split('/')[-3].split(' ')[-1]
            view = patient + '_' + file.split('/')[-2].replace(' ', '')
            # Make high risk if positive for cancer
            cancer = 1
        else:
            patient = 'BRCAneg_' + file.split('/')[-3].split('_')[-1] + '_' + file.split('/')[-2]
            view = patient + '_' + file.split('/')[-1].split('.')[0][-1]
            cancer = 0

        # Load and resize image
        try: image, accno, shape, _, _ = sdl.load_DICOM_2D(file)
        except:
            print ("Failed to load: ", file)
            continue

        """
                We have two methods to generate breast masks, they fail on different examples. 
                Use method 1 and if it generates a mask with >80% of pixels masked on < 10% we know it failed
                So then use method 2
        """
        mask = sdl.create_mammo_mask(image, check_mask=True)

        # Some masks just won't play ball
        mask_idx = np.sum(mask) / (image.shape[0] * image.shape[1])
        if mask_idx > 0.8 or mask_idx < 0.1:
            print('Failed to generate mask... ', view)
            continue

        # Multiply image by mask to make background 0
        try: image *= mask
        except:
            print ('Failed to apply mask... ', view, image.shape, image.dtype, mask.shape, mask.dtype)

        # Resize and generate label mask. 0=background, 1=no cancer, 2 = cancer
        image = sdl.zoom_2D(image, [box_dims, box_dims])
        labels = sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims]).astype(np.uint8) * (cancer + 1)

        # Normalize the mammograms using contrast localized adaptive histogram normalization
        image = sdl.adaptive_normalization(image).astype(np.float32)

        # Zero the background again.
        image *= sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims])

        # Save the data
        data[index] = {'data': image, 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': group, 'patient': patient, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1
        del image, mask, labels

    # Done with all patients
    print('Made %s BRCA boxes from %s patients' % (index, pt,), counter)

    # Save the data.
    sdl.save_dict_filetypes(data[0])
    sdl.save_segregated_tfrecords(4, data, 'patient', 'data/BRCA')


def pre_process_RISK(box_dims=1024):

    """
    Loads the files to a protobuf
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, risk_dir)
    shuffle(filenames)

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0], {}, {}, 0, 0

    for file in filenames:

        """
        Retreive patient number...

        # /RiskStudy/LowRisk/Normal/1/imgxxx.dcm (1+)

        We want: group = source of positive, brca vs risk 
        Patient = similar to accession, (can have multiple views) (RiskHigh_Cancer_1, RiskNl_Normal_1)
        View = unique to that view
        Label = 1 if cancer, 0 if not 
        """

        group = 'RISK'

        patient = file.split('/')[-4] + '_' +  file.split('/')[-3] + '_' + file.split('/')[-2]
        view = patient + '_' + file.split('/')[-1].split('.')[0][-1]

        if 'Cancer' in file: cancer = 1
        else: cancer = 0

        # Load and resize image
        try:
            image, accno, shape, _, _ = sdl.load_DICOM_2D(file)
        except:
            print("Failed to load: ", file)
            continue

        """
                We have two methods to generate breast masks, they fail on different examples. 
                Use method 1 and if it generates a mask with >80% of pixels masked on < 10% we know it failed
                So then use method 2
        """
        mask = sdl.create_mammo_mask(image, check_mask=True)

        # Some masks just won't play ball
        mask_idx = np.sum(mask) / (image.shape[0] * image.shape[1])
        if mask_idx > 0.8 or mask_idx < 0.1:
            print('Failed to generate mask... ', view)
            continue

        # Multiply image by mask to make background 0
        image *= mask

        # Resize and generate label mask. 0=background, 1=no cancer, 2 = cancer
        image = sdl.zoom_2D(image, [box_dims, box_dims])
        labels = sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims]).astype(np.uint8) * (cancer + 1)

        # Normalize the mammograms using contrast localized adaptive histogram normalization
        image = sdl.adaptive_normalization(image).astype(np.float32)

        # Zero the background again.
        image *= sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims])

        # Save the data
        data[index] = {'data': image, 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': group, 'patient': patient, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1
        del image, mask, labels

    # Done with all patients
    print('Made %s RISK boxes from %s patients' % (index, pt,), counter)

    # Save the data.
    sdl.save_dict_filetypes(data[0])
    sdl.save_segregated_tfrecords(4, data, 'patient', 'data/RISK')


def pre_process_CALCS(box_dims=1024):

    """
    Loads the files to a protobuf
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, calc_dir)
    shuffle(filenames)

    # Don't include folder 3 or 4 dicoms
    filenames = [x for x in filenames if '/1/' in x or '/2/' in x]

    # Global variables
    display, counter, data, index, pt = [], [0, 0], {}, 0, 0

    for file in filenames:

        """
        Retreive patient number
        
        # /Calcs/Eduardo/ADH/Patient 25 YES/1/ser42096img00006.dcm'

        We want: group = source of positive, brca vs risk 
        Patient = similar to accession, (can have multiple views) (CALCSADH_19_YES)
        View = unique to that view (CALCSADH_19_YES_CC)
        Label = 1 if cancer, 0 if not 
        """

        group = 'CALCS'

        patient = 'CALCS' + file.split('/')[-4] + '_' + file.split('/')[-3].split(' ')[1] + '_' + file.split('/')[-1].split('.')[0][-1]
        view = patient + '_' + ('CC' if '/1/' in file else 'MLO')

        # All of these are technically cancer
        cancer = 1

        # Load and resize image
        try:
            image, accno, shape, _, _ = sdl.load_DICOM_2D(file)
        except:
            print("Failed to load: ", file)
            continue

        """
        We have two methods to generate breast masks, they fail on different examples. 
        Use method 1 and if it generates a mask with >80% of pixels masked on < 10% we know it failed
        So then use method 2
        """
        mask = sdl.create_mammo_mask(image, check_mask=True)

        # Some masks just won't play ball
        mask_idx = np.sum(mask) / (image.shape[0] * image.shape[1])
        if mask_idx > 0.8 or mask_idx < 0.1:
            print('Failed to generate mask... ', view)
            continue

        # Multiply image by mask to make background 0
        image *= mask

        # Resize and generate label mask. 0=background, 1=no cancer, 2 = cancer
        image = sdl.zoom_2D(image, [box_dims, box_dims])
        labels = sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims]).astype(np.uint8) * (cancer+1)

        # Normalize the mammograms using contrast localized adaptive histogram normalization
        image = sdl.adaptive_normalization(image).astype(np.float32)

        # Zero the background again.
        image *= sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims])

        # Save the data
        data[index] = {'data': image, 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': group, 'patient': patient, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1
        del image, mask, labels

    # Done with all patients
    print ('Made %s CALCS boxes from %s patients' %(index, pt,))

    # Save the data. Saving with size 2 goes sequentially
    sdl.save_dict_filetypes(data[0])
    sdl.save_segregated_tfrecords(4, data, 'patient', 'data/CALCS')


def pre_process_PREV(box_dims=1024):

    """
    Loads the chemoprevention
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, chemo_dir)
    shuffle(filenames)

    # Include the baseline and follow up studies
    filenames = [x for x in filenames if '#' not in x]

    # Global variables
    display, counter, data, index, pt = [], [0, 0], {}, 0, 0

    for file in filenames:

        """
        Retreive patient number

        # Chemoprevention/Treated/Patient 2/L CC 5YR/xxx.dcm

        We want: group = source of positive, brca vs risk 
        Patient = similar to accession, (can have multiple views) (CALCSADH_19_YES)
        View = unique to that view (CALCSADH_19_YES_CC)
        Label = 1 if cancer, 0 if not 
        """

        group = 'PREV'
        patient = 'PREV' + file.split('/')[-4] + '_' + file.split('/')[-3].split(' ')[1] + '_'
        if '5YR' in file: patient += '5YR'
        else: patient += 'BASE'
        view = patient + '_' + file.split('/')[-2].replace(' ', '')[:3]

        # All of these are technically cancer
        cancer = 1

        # Load and resize image
        try:
            image, accno, shape, _, _ = sdl.load_DICOM_2D(file)
        except:
            print("Failed to load: ", file)
            continue

        """
        We have two methods to generate breast masks, they fail on different examples. 
        Use method 1 and if it generates a mask with >80% of pixels masked on < 10% we know it failed
        So then use method 2
        """
        mask = sdl.create_mammo_mask(image, check_mask=True)

        # Some masks just won't play ball
        mask_idx = np.sum(mask) / (image.shape[0] * image.shape[1])
        if mask_idx > 0.8 or mask_idx < 0.1:
            print('Failed to generate mask... ', view)
            continue

        # Multiply image by mask to make background 0
        image *= mask

        # Resize and generate label mask. 0=background, 1=no cancer, 2 = cancer
        image = sdl.zoom_2D(image, [box_dims, box_dims])
        labels = sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims]).astype(np.uint8) * (cancer + 1)

        # Normalize the mammograms using contrast localized adaptive histogram normalization
        image = sdl.adaptive_normalization(image).astype(np.float32)

        # Zero the background again.
        image *= sdl.zoom_2D(mask.astype(np.int16), [box_dims, box_dims])

        # Save the data
        data[index] = {'data': image, 'label_data': labels, 'file': file, 'shapex': shape[0], 'shapy': shape[1],
                       'group': group, 'patient': patient, 'view': view, 'cancer': cancer, 'accno': accno}

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1
        del image, mask, labels

    # Done with all patients
    print('Made %s Chemoprevention boxes from %s patients' % (index, pt,))

    # Save the data.
    sdl.save_segregated_tfrecords(4, data, 'patient', 'data/PREV')


# Load the protobuf
def load_protobuf(training=True):

    """
    Loads the protocol buffer into a form to send to shuffle
    """

    # Define filenames
    if training:
        num_files = len(sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir))
        files = tf.data.Dataset.list_files(os.path.join(FLAGS.data_dir, '*.tfrecords'))
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=num_files,
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        files = sdl.retreive_filelist('tfrecords', False, path=FLAGS.data_dir)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=1)

    print('******** Loading Files: ', files)

    # Shuffle and repeat if training phase
    if training:

        # Buffer size only effect randomness of shuffle not samples used. Pick epoch size to be perfect or 10k
        dataset = dataset.shuffle(buffer_size=FLAGS.epoch_size)
        # Repeat duplicates the dataset x times
        dataset = dataset.repeat(2)

    """
        Functions to map the dataset
        - Define the first map function, can just use lambda instead in the map call
        - Parse the tfrecord into tensors
        - Fuse the mapping and batching
        - Map the data set with preprocessing steps
        """

    _parse_function = lambda dataset: sdl.load_tfrecords(dataset, [FLAGS.box_dims, FLAGS.box_dims], tf.float32,
                           segments='label_data', segments_dtype=tf.uint8, segments_shape=[FLAGS.box_dims, FLAGS.box_dims])
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    scope = 'data_augmentation' if training else 'input'
    with tf.name_scope(scope): dataset = dataset.map(DataPreprocessor(training), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    """
        - If the user-defined function passed into the map transformation is expensive,
            apply the cache transformation after the map transformation as long as the resulting dataset
            can still fit into memory or local storage.
        - Batch the dataset and drop remainder. Can try batch before map if map is small
        - Prefetch batches to start preprocessing the next batch before the training step is done
        """

    # dataset = dataset.cache()
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Make an initializable iterator
    iterator = dataset.make_initializable_iterator()

    # Retreive the batch
    examples = iterator.get_next()

    # Return data as a dictionary
    return examples, iterator


class DataPreprocessor(object):

    # Applies transformations to dataset

  def __init__(self, distords):

    self._distords = distords

  def __call__(self, data):

    if self._distords:  # Training

        # Expand dims by 1
        data['data'] = tf.expand_dims(data['data'], -1)
        data['label_data'] = tf.expand_dims(data['label_data'], -1)

        #Random rotate
        angle = tf.random_uniform([], -0.45, 0.45)
        data['data'] = tf.contrib.image.rotate(data['data'], angle, interpolation='BILINEAR')
        data['label_data'] = tf.contrib.image.rotate(data['label_data'], angle, interpolation='NEAREST')

        # Random shear:
        # rand = []
        # for z in range(4):
        #     rand.append(tf.random_uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32))
        # data['data'] = tf.contrib.image.transform(data['data'], [1, rand[0], rand[1], rand[2], 1, rand[3], 0, 0])
        # data['label_data'] = tf.contrib.image.transform(data['label_data'], [1, rand[0], rand[1], rand[2], 1, rand[3], 0, 0])

        # Reshape, bilinear for labels, cubic for data
        data['data'] = tf.image.resize_images(data['data'], [FLAGS.network_dims, FLAGS.network_dims],
                                              tf.compat.v1.image.ResizeMethod.BICUBIC)
        data['label_data'] = tf.image.resize_images(data['label_data'], [FLAGS.network_dims, FLAGS.network_dims],
                                                    tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Randomly flip
        def flip(mode=None):

            if mode == 1:
                img, lbl = tf.image.flip_up_down(data['data']), tf.image.flip_up_down(data['label_data'])
            elif mode == 2:
                img, lbl = tf.image.flip_left_right(data['data']), tf.image.flip_left_right(data['label_data'])
            else:
                img, lbl = data['data'], data['label_data']
            return img, lbl

        # Maxval is not included in the range
        data['data'], data['label_data'] = tf.cond(tf.squeeze(tf.random.uniform([1], 0, 2, dtype=tf.int32)) > 0, lambda: flip(1), lambda: flip(0))
        data['data'], data['label_data'] = tf.cond(tf.squeeze(tf.random.uniform([1], 0, 2, dtype=tf.int32)) > 0, lambda: flip(2), lambda: flip(0))

        # Random contrast and brightness
        data['data'] = tf.image.random_brightness(data['data'], max_delta=2)
        data['data'] = tf.image.random_contrast(data['data'], lower=0.975, upper=1.025)

        # Random gaussian noise
        T_noise = tf.random.uniform([], 0, 0.1)
        noise = tf.random.uniform(shape=[FLAGS.network_dims, FLAGS.network_dims, 1], minval=-T_noise, maxval=T_noise)
        data['data'] = tf.add(data['data'], tf.cast(noise, tf.float32))


    else: # Testing

        # Expand dims by 1
        data['data'] = tf.expand_dims(data['data'], -1)
        data['label_data'] = tf.expand_dims(data['label_data'], -1)

        # Reshape, bilinear for labels, cubic for data
        data['data'] = tf.image.resize_images(data['data'], [FLAGS.network_dims, FLAGS.network_dims],
                                              tf.compat.v1.image.ResizeMethod.BICUBIC)
        data['label_data'] = tf.image.resize_images(data['label_data'], [FLAGS.network_dims, FLAGS.network_dims],
                                                    tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR)

    return data