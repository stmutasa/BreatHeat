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
import shutil

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


def pre_process_1k(box_dims=1024):
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

        # Skip non breasts
        if 'BREAST' not in header['tags'].BodyPartExamined: continue

        # Skip mag views based on SOD/SID
        SOD = header['tags'].DistanceSourceToPatient
        SID = header['tags'].DistanceSourceToDetector
        SOD_SID = int(SID) / int(SOD)
        if SOD_SID > 1.25: continue

        # Skip views that aren't CC or MLO
        if view != 'MLO' and view != 'CC' and view != 'XCCL': continue

        # Set info
        patient = '1k_' + MRN + '_' + accno
        view = '1k_' + accno + '_' + laterality + view
        group = '1k'
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

    # Done with all patients
    print('Made %s BRCA boxes from %s patients' % (index, pt,), counter)

    # Save the data.
    sdl.save_dict_filetypes(data[0])
    sdl.save_segregated_tfrecords(2, data, 'patient', 'data/1k')


def re_save_1k():
    """
    Loads the 1k chemoprevention files and resaves the DICOM
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Julia1k'
    save_path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Reprocessed_1k/'
    filenames = sdl.retreive_filelist('**', True, path)
    shuffle(filenames)

    # Global variables
    display, counter, skipped, data_test, index, pt = [], [0, 0], [], {}, 0, 0

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
        except Exception as e:
            # print('DICOM Error: %s' %e)
            continue

        # Retreive the accession number
        try:
            view = header['tags'].ViewPosition
            laterality = header['tags'].ImageLaterality
        except Exception as e:
            # print('Header error: %s' %e)
            continue

        # Resize image
        image = sdl.zoom_2D(image, [256, 256]).astype(np.int16)

        """
            Some Mag views are still getting through
            Also some negative photometrics are getting through
                SID/SOD skips most
                FieldOfViewDimensions? 
                Tried and failed: DetectorBinning FocalSpots Grid np.max()
                ViewPosition not CC or MLO
        """

        # Skip non breasts
        if 'BREAST' not in header['tags'].BodyPartExamined:
            test1 = header['tags'].BodyPartExamined
            image = sdd.return_image_text_overlay(('%s %s' % ('NOT BREAST', test1)), image)
            skipped.append(image)
            continue

        # Skip mag views based on SOD/SID
        SOD = header['tags'].DistanceSourceToPatient
        SID = header['tags'].DistanceSourceToDetector
        SOD_SID = int(SID) / int(SOD)
        if SOD_SID > 1.25:
            test1 = SOD_SID
            image = sdd.return_image_text_overlay(('%s %s' % ('SOD_SID: ', test1)), image)
            skipped.append(image)
            continue

        # Skip views that aren't CC or MLO
        if view != 'MLO' and view != 'CC' and view != 'XCCL':
            image = sdd.return_image_text_overlay(('%s %s' % ('View: ', view)), image)
            skipped.append(image)
            continue

        # Save display
        # test1 = header['tags'].Grid
        # test2 = header['tags'].DetectorBinning
        # image = sdd.return_image_text_overlay(('%s %s' % (test2, test1)), image)
        # display.append(image)
        del image

        # Set info
        patient = '1k_' + str(index)
        view = patient + '_' + laterality + view

        # Filename
        savedir = (save_path + accno + '/')
        savefile = savedir + view + '.dcm'
        if not os.path.exists(savedir): os.makedirs(savedir)

        # Copy to the destination folder
        shutil.copyfile(file, savefile)
        if index % 100 == 0 and index > 1:
            print('\nSaving pt %s of %s to dest: %s_%s\n' % (index, len(filenames), accno, view))

        # Increment counters
        index += 1
        pt += 1

    print('Done with %s images saved' % index)
    sdd.display_volume(skipped, True)
    # sdd.display_volume(display, True)


pre_process_1k()
# re_save_1k()
