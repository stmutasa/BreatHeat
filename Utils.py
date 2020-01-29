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
            print('DICOM Error: %s' % e)
            continue

        # Retreive the view
        try:
            view = header['tags'].ViewPosition
        except Exception as e:
            print('Header error: %s' % e)
            continue

        # Retreive the Laterality
        try:
            laterality = header['tags'].ImageLaterality
        except:
            try:
                laterality = header['tags'].Laterality
            except Exception as e:
                print('Header error: %s' % e)
                continue

        """
            Some Mag views are still getting through
            Also some negative photometrics are getting through
                SID/SOD skips most
                FieldOfViewDimensions? 
                Tried and failed: DetectorBinning FocalSpots Grid np.max()
                ViewPosition not CC or MLO
        """

        # Skip non breasts
        if 'BREAST' not in header['tags'].BodyPartExamined: continue

        # Skip mag views based on SOD/SID
        SOD = header['tags'].DistanceSourceToPatient
        SID = header['tags'].DistanceSourceToDetector
        SOD_SID = int(SID) / int(SOD)
        if SOD_SID > 1.25: continue

        # CC Only!!
        if view != 'CC' and view != 'XCCL': continue

        # Set info
        patient = '1YR_' + str(index)
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


re_save_1k()
