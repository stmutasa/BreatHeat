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
short_fu = risk_dir + '1yr_FU/Raw/'
brca_dir = home_dir + 'BRCA/'
calc_dir = home_dir + 'Calcs/Eduardo/'
chemo_dir = home_dir + 'Chemoprevention/'

sdl = SDL.SODLoader(data_root=home_dir)
sdd = SDD.SOD_Display()


def re_save_1yr(type='CC'):
    """
    Loads the 1 year followup chemoprevention files and resaves the CC and/or MLO DICOM
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/RiskStudy/1yr_FU/Raw/'
    save_path_CC = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/RiskStudy/1yr_FU/Processed_CC/'
    save_path_All = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/RiskStudy/1yr_FU/Processed_MLO_CC/'
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
        if type == 'CC':
            if view != 'CC' and view != 'XCCL': continue
        else:
            if view != 'CC' and view != 'XCCL' and view != 'MLO': continue

        # Set info
        patient = '1YR_' + str(index)
        view = patient + '_' + laterality + view

        # Filename
        if type == 'CC':
            savedir = (save_path_CC + accno + '/')
        else:
            savedir = (save_path_All + accno + '/')
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


def re_save_adjuvant(type='CC'):
    """
    Loads the Adjuvant treatment files and resaves the CC and/or MLO DICOM
    :param box_dims: dimensions of the saved images
    :return:
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Adjuvant/Raw/'
    save_path_CC = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Adjuvant/Processed_CC/'
    save_path_All = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Adjuvant/Processed_MLO_CC/'
    filenames = sdl.retreive_filelist('**', True, path)

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
        try:
            SOD = header['tags'].DistanceSourceToPatient
            SID = header['tags'].DistanceSourceToDetector
            SOD_SID = int(SID) / int(SOD)
        except:
            continue
        if SOD_SID > 1.25: continue

        # CC Only!!
        if type == 'CC':
            if view != 'CC' and view != 'XCCL': continue
        else:
            if view != 'CC' and view != 'XCCL' and view != 'MLO': continue

        # Set info
        patient = 'ADJ_' + str(index)
        view = patient + '_' + laterality + view

        # Filename
        if type == 'CC':
            savedir = (save_path_CC + accno + '/')
        else:
            savedir = (save_path_All + accno + '/')
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


def save_date_adj():
    """
    Saves the date of each adjuvant file into an accno key dict
    """

    # Load the filenames and randomly shuffle them
    path = home_dir + 'Adjuvant/Processed_CC/'
    filenames = sdl.retreive_filelist('**', True, path)

    # Global variables
    data = {}
    index = 0

    for file in filenames:

        # Load the Dicom
        try:
            header = sdl.load_DICOM_Header(file, multiple=False)
            accno = file.split('/')[-2]
            series = str(header['tags'].SeriesDescription)
        except:
            continue

        # Skip non breasts
        if 'BREAST' not in header['tags'].BodyPartExamined: continue

        # Skip mag views based on SOD/SID
        try:
            SOD = header['tags'].DistanceSourceToPatient
            SID = header['tags'].DistanceSourceToDetector
            SOD_SID = int(SID) / int(SOD)
        except:
            continue
        if SOD_SID > 1.25: continue

        # Get date
        try:
            date = str(header['tags'].AcquisitionDate)
        except:
            try:
                date = str(header['tags'].ContentDate)
            except:
                date = str(header['tags'].SeriesDate)
        if not date: date = str(header['tags'].SeriesDate)

        data[accno] = {'date': date, 'desc': series}
        index += 1
        if index % 1000 == 0: print(index, ' done.')

    sdl.save_Dict_CSV(data, 'dates.csv')
    print('Done with %s images saved' % len(data))


def save_date_risk():

    """
    Saves the date of each positive RISK file into an accno key dict
    """

    # Load the filenames and randomly shuffle them
    path = risk_dir
    filenames = sdl.retreive_filelist('**', True, path)

    # Global variables
    data = {}
    index = 0

    for file in filenames:

        # Load the Dicom
        try:
            header = sdl.load_DICOM_Header(file, multiple=False)
            accno = header['tags'].StudyID
            series = str(header['tags'].SeriesDescription)
        except:
            continue

        # Get date
        try:
            date = str(header['tags'].AcquisitionDate)
        except:
            try:
                date = str(header['tags'].ContentDate)
            except:
                date = str(header['tags'].SeriesDate)
        if not date: date = str(header['tags'].SeriesDate)

        data[accno] = {'date': date, 'desc': series}
        index += 1
        if index % 1000 == 0: print(index, ' done.')

    sdl.save_Dict_CSV(data, 'Risk_dates.csv')
    print('Done with %s images saved' % len(data))


def eval_DLs():

    """
    Checks the missing files and sees whats up
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Adjuvant/Raw/'
    root_dir = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/'
    missings = sdl.load_CSV_Dict('ACC', root_dir + 'Adjuvant_Missing.csv')
    filenames = sdl.retreive_filelist('*', True, path)

    # Global variables
    display, counter, skipped, data_test, index, pt = [], [0, 0], [], {}, 0, 0

    # Loop through all the "missing" files
    for acc, dic in missings.items():

        # trackers
        foex, fiex = False, False

        # Make sure the folder even exists
        for file in filenames:
            rt = os.path.basename(file)
            if acc == rt:
                foex = True
                break

        if foex != True:
            #print ('%s on MRN %s has no folder' %(acc, dic['MRN']))
            acc = [int(s) for s in acc if s.isdigit()]
            acc = ''.join(map(str, acc))
            # print (acc)
            continue

        # Make sure folder isn't empty
        subfiles = sdl.retreive_filelist('**', True, file)
        if len(subfiles) == 0:
            #print (acc)
            continue

        # Print the study description
        for sfs in subfiles:
            try:
                hdr = sdl.load_DICOM_Header(sfs, False)
                sdesc = hdr['tags'].StudyDescription
                if not sdesc: continue
                # Filter down to outside studies only
                if 'OUTSIDE' not in sdesc: continue
                if 'MR' in sdesc or 'US' in sdesc: continue
                mfc = hdr['tags'].Manufacturer
                if not mfc: continue
                # filter down to non GE outside studies
                if 'Philips' not in mfc and 'SIEMENS' not in mfc: continue
                # Load the Dicom
                try: _ = sdl.load_DICOM_2D(file)
                except Exception as e:
                    print('DICOM Error: %s' % e)
                    continue
                print (acc, ',', mfc)
                #break
            except:
                continue


def check_new(vtype='CC'):

    """
    Checks the missing files and sees whats up
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/ADJ2/'
    filenames = sdl.retreive_filelist('**', True, path)

    # Global variables
    display, counter, skipped, data, dataf, index, pt = [], [0, 0], [], {}, {}, 0, 0

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
            #print('DICOM Error: %s' % e)
            dataf[index] = {'Accno': file.split('/')[-2], 'info': e}
            index += 1
            continue

        # Retreive the view
        try:
            view = header['tags'].ViewPosition
        except Exception as e:
            #print('Header error: %s' % e)
            dataf[index] = {'Accno': accno, 'info': e}
            index += 1
            continue

        # Retreive the Laterality
        try:
            laterality = header['tags'].ImageLaterality
        except:
            try:
                laterality = header['tags'].Laterality
            except Exception as e:
                #print('Header error: %s' % e)
                dataf[index] = {'Accno': accno, 'info': e}
                index += 1
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
        if 'BREAST' not in header['tags'].BodyPartExamined:
            dataf[index] = {'Accno': accno, 'info': header['tags'].BodyPartExamined}
            index += 1
            continue

        # Skip mag views based on SOD/SID
        try:
            SOD = header['tags'].DistanceSourceToPatient
            SID = header['tags'].DistanceSourceToDetector
            SOD_SID = int(SID) / int(SOD)
        except:
            dataf[index] = {'Accno': accno, 'info': 'No SOD_SID'}
            index += 1
            continue
        if SOD_SID > 1.25:
            dataf[index] = {'Accno': accno, 'info': SOD_SID}
            index += 1
            continue

        # CC Only!!
        if vtype == 'CC':
            if view != 'CC' and view != 'XCCL':
                dataf[index] = {'Accno': accno, 'info': view}
                index += 1
                continue
        else:
            if view != 'CC' and view != 'XCCL' and view != 'MLO': continue

        # Set info
        pat = accno + '_' + laterality + view
        data[index] = pat

        if accno not in display: display.append(accno)
        skipped.append(pat)
        print (pat)

        # Increment counters
        index += 1
        pt += 1

    # Un-fail stage
    print ('Removing fake failures')
    dataf2 = dict(dataf)
    for key, index in data.items():
        for keyf, indexf in dataf.items():
            if indexf['Accno'] == index.split('_')[0]:
                if keyf in dataf2: del dataf2[keyf]


    print('Done with %s patients saved. %s failure are now %s' % (len(display), len(dataf), len(dataf2)))
    sdl.save_Dict_CSV(data, 'Saved.csv')
    sdl.save_Dict_CSV(dataf2, 'failed.csv')
    print ('K')


def check_outside():

    """
    Checks the missing files and sees whats up
    """

    # Load the filenames and randomly shuffle them
    path = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/Adjuvant/Raw/'
    root_dir = '/media/stmutasa/Slow1/PycharmProjects/Datasets/BreastData/Mammo/'
    missings = sdl.load_CSV_Dict('ACC', root_dir + 'Adjuvant_Missing.csv')
    filenames = sdl.retreive_filelist('*', True, path)

    # Global variables
    display, counter, skipped, data_test, index, pt = [], [0, 0], [], {}, 0, 0

    # Loop through all the "missing" files
    for acc, dic in missings.items():

        # trackers
        foex, fiex = False, False

        # Make sure the folder even exists
        for file in filenames:
            rt = os.path.basename(file)
            if acc == rt:
                foex = True
                break

        if foex != True:
            continue

        # Make sure folder isn't empty
        subfiles = sdl.retreive_filelist('**', True, file)
        if len(subfiles) == 0:
            continue

        # Print the study description
        for sfs in subfiles:
            try:
                hdr = sdl.load_DICOM_Header(sfs, False)
                sdesc = hdr['tags'].StudyDescription
                if not sdesc: continue
                # Filter down to outside studies only
                if 'OUTSIDE' not in sdesc: continue
                if 'MR' in sdesc or 'US' in sdesc: continue
                # Load the Dicom
                try: image, accno, _, _, hdr = sdl.load_DICOM_2D(sfs)
                except Exception as e:
                    print('DICOM Error: %s' % e, sfs)
                    continue

                # Now only dealing with outside films with images
                savename = 'data/saved/' + accno + '_' + str(index) + '.jpg'
                savedcm = savename.replace('.jpg', '.dcm')

                # Save the image
                sdl.save_image(image, savename)
                shutil.copyfile(sfs, savedcm)

                # Increment
                index += 1
            except:
                continue


#save_date_adj()
#save_date_risk()
#eval_DLs()
#check_new()
#check_outside()
