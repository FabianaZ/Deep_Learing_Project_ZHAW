### Import section
from PIL import Image
import pandas as pd
import os
import shutil
import numpy as np
import random


def create_dataset(path_to_500hPaGeopotential_folder, labeling_csv_file):
    parent_folder = path_to_500hPaGeopotential_folder + '\\'
    labeling_file = labeling_csv_file

    print('Creating dataset: ', labeling_file[:-4])

    # read in label files
    labels = pd.read_csv(parent_folder + 'Labeling\\' + labeling_file, sep=',')
    labels.drop(columns=['WindSpeed'], inplace=True)
    labels.set_index('TimeStamp', inplace=True)
    labels.index = pd.to_datetime(labels.index)
    print('\tLabels collected:', end='\t')
    for label in labels['WindLabel'].unique():
        print(label, end='\t')
    print('')

    # Create list with the set assignments
    fractions = {'test': 0.2, 'train': 0.6, 'validation': 0.2}  # must sum up to 1.0
    num_of_labels = len(labels)
    set_assignment = np.zeros(len(labels.index))

    set_assignment = []
    for set in fractions:
        list_of_strings = [set] * round(num_of_labels * fractions[set])
        set_assignment += list_of_strings
    while len(set_assignment) < num_of_labels:
        set_assignment += fractions[1]
        print('\tAdded...')
    if len(set_assignment) > num_of_labels:
        set_assignment = set_assignment[0, num_of_labels]
        print('\tReduced...')
    random.shuffle(set_assignment)

    # Create dataset folder if necessary
    directory_for_data_sets = parent_folder + 'Datasets'
    if not os.path.exists(directory_for_data_sets):
        os.makedirs(directory_for_data_sets)
        print('\tCreating Dataset Directory')

    # Remove existing sub folders and create a new one for every label
    sub_folder_for_data_sets = directory_for_data_sets + '\\' + labeling_file[:-4]
    if os.path.exists(sub_folder_for_data_sets):
        shutil.rmtree(sub_folder_for_data_sets)
    os.makedirs(sub_folder_for_data_sets)

    for set in fractions:
        for label in labels['WindLabel'].unique():
            os.makedirs(sub_folder_for_data_sets + '\\' + str(set) + '\\' + str(label))
            print('\tCreating Subfolder:\t', str(set), str(label))

    path_to_original_data = parent_folder + '\\original_data'
    file_count = {}
    for set in fractions:
        file_count[set] = 0

    count = 0
    for index, row in labels.iterrows():
        lab = row['WindLabel']  # do we have missing labels?
        date = index.strftime('%Y%m%d')
        year = index.strftime('%Y')

        # generate potential path to image
        image_name = 'geopotential_' + date + '.gif'
        file_path = path_to_original_data + '\\' + year + '\\' + image_name

        # generate target path
        target_folder = sub_folder_for_data_sets + '\\' + str(set_assignment[count]) + '\\' + str(lab)

        # try to find corresponding images and copy them to the labeled folder
        try:
            shutil.copy(file_path, target_folder)
            file_count[str(set_assignment[count])] += 1
        except:
            print('\tFile not found:\t' + image_name)
        count += 1
    print('\tFiles moved to corresponding folder:\t', file_count)


def get_image_paths(path_to_dataset):
    # find image paths
    image_paths = []
    for root, dirs, files in os.walk(path_to_data_set, topdown=False):
        for file in files:
            # only use .gif files
            if file[-4:] == '.gif':
                file_path = os.path.join(root, file)
                image_paths += [file_path]
    return image_paths


def process_images(path_to_data_set, crop=True, grey=False, resize=False, black_white=True):
    image_paths = get_image_paths(path_to_data_set)

    # Image transform parameters
    crop_size = (100, 100, 500, 350)
    resize_pixel = (100, 100)

    new_size = (crop_size[2]-crop_size[1], crop_size[3]-crop_size[0])

    # Print processing inputs
    print('Image processing....')
    print('\tOriginal image size:', end='\t')
    print(Image.open(image_paths[0]).size)


    if crop:
        print('\tImage cropped to:\t', new_size)
    if black_white:
        print('\tImage transformed to:\tBlack and White')
    if grey:
        print('\tImage transformed to:\tGreyscale')
    if resize:
        print('\tImage resized to:\t', resize_pixel, 'Pixel')



    counter = 0
    for path in image_paths:
        try:
            with Image.open(path) as im:
                im = im.convert('RGB')

                if crop:
                    im = im.crop(crop_size)
                if black_white:
                    thresh = 240
                    fn = lambda x: 255 if x > thresh else 0
                    im = im.convert('L').point(fn, mode='1')
                if grey:
                    im = im.convert('L')
                if resize:
                    im = im.resize(resize_pixel)
                if counter % 500 == 0:
                    print('\tProcessed: ', counter, '/', len(image_paths))



                counter += 1
                im.save(path)
        except:
            print('\tImage Error:\t', path[-25:])
    #show last image
    #im.show()

    print('\tOutput Images consist of:\t', len(np.asarray(im).flatten()), 'Pixel')



#################
### MAIN CODE ###
#################

# Code needs following folderstructure

# 500hPaGeopotential
# ¦ --- Labeling
#           ¦ --- labeling_csv_file.csv
# ¦ --- original_data
#           ¦ --- 1995 Folder with images
#           ¦ --- 1996 Folder with images
#           ¦ --- ...


# path to parent project folder
parent_folder = "C:\\Users\\Dominik Wolf\\Desktop\\DL_Project\\500hPaGeopotential"  # [TO BE ADPATED ON YOUR LOCAL MACHINE]

# csv file in the Labeling Folder
labeling_file = 'binary_small.csv'  # [CHOOSE A LABELING FILE WITHIN THE LABELING FOLDER]

# creating folder structure and data set
create_dataset(parent_folder, labeling_file)

path_to_data_set = parent_folder + '\\Datasets\\' + str(labeling_file[:-4])
process_images(path_to_data_set)
