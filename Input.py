"""
Does our loading and preprocessing of files to a protobuf

Who are the patients who have cancer in our dataset: (1885 / 1824  no cancer views, 356 / 356 (+407 Ca+) cancer views)

BRCA:
	(37*4 or 146) BRCA/G1_PosBRCA/Cancer/Patient 2/R CC/xxx.dcm (4 views each patient)
	TODO: What breast gets the cancer? Are these pre-cancer scans

Calcs:
    TODO: Find out if there are CC and MLO views
    File 1 and File 2 contain CC and MLO views of the affected breast - These pts have cancer though
    The /new data does not contain full field mammograms - dammit man!
	Invasive
	Microinvasion
	(ADH and DCIS is not cancer)

Chemoprevention:
	(All just high risk ADH, LCIS and DCIS, no cancers)

RiskStudy:
	(201*2) RiskStudy/HighRisk/Cancer/4/imgxx.dcm (2 views each pt.)
	(Low risk is low risk with no cancer)
	(High Risk / Normal didnt get cancer)
"""

from matplotlib import pyplot as plt

import glob

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD
from pathlib import Path

from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/BreastData/Mammo/'
risk_dir = home_dir + 'RiskStudy/'
brca_dir = home_dir + 'BRCA/'
calc_dir = home_dir + 'Calcs/Eduardo/'

Cancer_count = [0, 0]

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
    #shuffle(filenames)

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
            if 'NoCancer' in file: cancer=0
            else: cancer = 1
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

        # TODO: Testing
        display.append(image)
        print(pt, '-- Patient: %s Label: %s Res: %s' % (view, cancer, image.shape))
        if pt > 19: break

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1
        del image, mask, labels

    # Done with all patients
    print('Made %s BRCA boxes from %s patients' % (index, pt,))

    # Save the data. TODO: Need to use a segregated shuffle on per pt basis
    sdl.save_tfrecords(data, 2, test_size=100, file_root='data/BRCA')

    # TODO: Testing
    sdd.display_volume(display, True)


def pre_process_RISK(box_dims=1024):
    """
    Loads the files to a protobuf
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, risk_dir)
    #shuffle(filenames)

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

        # TODO: Testing
        display.append(image)
        print(pt, '-- Patient: %s Label: %s Res: %s' % (view, cancer, image.shape))
        if pt > 19: break

        # Increment counters
        index += 1
        counter[cancer] += 1
        pt += 1
        del image, mask, labels

    # Done with all patients
    print('Made %s RISK boxes from %s patients' % (index, pt,))

    # Save the data. TODO: Need to use a segregated shuffle on per pt basis
    sdl.save_tfrecords(data, 2, test_size=100, file_root='data/RISK')

    # TODO: Testing
    sdd.display_volume(display, True)


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
    sdl.save_tfrecords(data, 2, test_size=100, file_root='data/CALCS')


def pre_process(box_dims=512):

    """
    Loads the files to a protobuf
    :param warps:
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    filenames = sdl.retreive_filelist('dcm', True, risk_dir)
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
    data = sdl.load_tfrecords(filenames, FLAGS.box_dims, tf.float32, channels=1, segments='label_data')

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


pre_process_BRCA()
pre_process_RISK()
pre_process_CALCS()
print ('\nCancer Counts: ', Cancer_count)