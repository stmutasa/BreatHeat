"""
Utilities, like loading our other test sets
"""

import numpy as np
import tensorflow as tf
import SODLoader as SDL
import SOD_Display as SDD
from pathlib import Path
import os
import pydicom as dicom

from random import shuffle

# Define the flags class for variables
FLAGS = tf.app.flags.FLAGS

# Define the data directory to use
home_dir = str(Path.home()) + '/PycharmProjects/Datasets/BreastData/Mammo/'

risk_dir = home_dir + 'RiskStudy/'
brca_dir = home_dir + 'BRCA/'
calc_dir = home_dir + 'Calcs/Eduardo/'
chemo_dir = home_dir + 'Chemoprevention/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()


def pre_process_Julia1k(box_dims=1024):
    """
    Loads the 1k chemoprevention files
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Julia1k'
    filenames = sdl.retreive_filelist('**', True, path)
    shuffle(filenames)

    # Global variables
    display, counter, data, data_test, index, pt = [], [0, 0], {}, {}, 0, 0

    for file in filenames:

        """
        Retreive patient number
        All of these are DICOMs
        View = unique to that view (BRCA_Cancer_1_LCC)
        Label = 1 if cancer, 0 if not 
        """

        # Load the Dicom
        try:
            image, accno, _, _, header = sdl.load_DICOM_2D(file)
            shape = image.shape
        except Exception as e:
            print('Unable to Load DICOM file: %s - %s' % (e, file))
            continue

        # Retreive the accession number
        try:
            MRN = header['tags'].PatientID
            view = header['tags'].ViewPosition
            laterality = header['tags'].ImageLaterality
        except Exception as e:
            print('Unable to Load header info: %s - %s' % (e, file))
            continue

        # Set info
        patient = 'Julia1k_' + MRN + '_' + accno
        view = patient + '_' + laterality + view
        group = 'Julia1k'
        cancer = 0

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
        try:
            image *= mask
        except:
            print('Failed to apply mask... ', view, image.shape, image.dtype, mask.shape, mask.dtype)

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

        # # Save after 3k
        # if index % 3000 == 0:
        #     sdl.sav

    # Done with all patients
    print('Made %s BRCA boxes from %s patients' % (index, pt,), counter)

    # Save the data.
    sdl.save_dict_filetypes(data[0])
    sdl.save_segregated_tfrecords(2, data, 'patient', 'data/Julia1k')


def check_files():
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Julia1k'

    DOMAIN = "http://localhost:5000"

    # The files
    files = sdl.retreive_filelist('**', True, path)
    test, accnos = [], []
    index, count = 0, 0
    save_dict, failed_dict = {}, {}

    print('Preprocessing %s files' % len(files))
    folders_ran = []
    for file in files:
        try:
            # Load accession number
            acc = get_accno(file)
            if acc == -1: continue

            # Get  unique counts
            folders_ran.append(os.path.dirname(file))
            unique, counts = np.unique(np.asarray(folders_ran), return_counts=True)
            check_dict = dict(zip(unique, counts))

            # File name
            patient = 'Julia1k_' + file.split('/')[-3]
            view = patient + '_' + str(check_dict[os.path.dirname(file)])
            test.append(open(file, 'rb'))
            accnos.append((patient, acc, view))
        except:
            print('Failed to load: ', file)
            continue


pre_process_Julia1k()
