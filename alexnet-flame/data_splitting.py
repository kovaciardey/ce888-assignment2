###
# The code in this file has been adapted from this tutorial
# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
###

from matplotlib import pyplot
from matplotlib.image import imread

import os
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

import shutil

old_dataset_location = "alexnet-other/modified-flame/"

old_fire_folder = "Training/Fire/"
old_no_fire_folder = "Training/No_Fire/"


# plot nine images from each class
def plot_nine(dataset, folder):
    for i in range(9):
        pyplot.subplot(330 + 1 + i)

        image = imread(dataset + folder + listdir(dataset + folder)[i])
        pyplot.imshow(image)

    pyplot.show()


# plot some data from both classes
# plot_nine(old_dataset_location, old_fire_folder)
# plot_nine(old_dataset_location, old_no_fire_folder)

dataset_home = 'alexnet-other/custom-flame/'
subdirs = ['train/', 'test/']

for subdir in subdirs:
    # create label subdirectories
    labeldirs = ['fire/', 'no_fire/']
    for labldir in labeldirs:
        newdir = dataset_home + subdir + labldir
        makedirs(newdir, exist_ok=True)

# if len(listdir(dataset_home + 'train-complete/fire/')) == 0:
#     counter = 0
#     for image in listdir(dataset_location + fire_folder):
#         old_location = dataset_location + fire_folder + image
#         new_location = dataset_home + 'train-complete/fire/' + str(counter) + '_fire.jpg'
#
#         shutil.move(old_location, new_location)
#
#         counter += 1


# if len(listdir(dataset_home + 'train-complete/no_fire')) == 0:
#     counter = 0
#     for image in listdir(dataset_location + no_fire_folder):
#         old_location = dataset_location + no_fire_folder + image
#         new_location = dataset_home + 'train-complete/no_fire/' + str(counter) + '_no-fire.jpg'
#
#         shutil.move(old_location, new_location)
#
#         counter += 1

dataset_location = 'alexnet-other/custom-flame/'
complete_fire_folder = 'train-complete/fire/'
complete_no_fire_folder = 'train-complete/no_fire/'

plot_nine(dataset_location, complete_fire_folder)
plot_nine(dataset_location, complete_no_fire_folder)

seed(3)
val_ratio = 0.25

sources = [dataset_location + complete_fire_folder, dataset_location + complete_no_fire_folder]

# change for the number of images ot take from each class
max_img = 10000

# split the training and test datasets
for source in sources:
    counter = 0
    for image in listdir(source):
        src = source + image

        dst_dir = 'train/'
        if random() < val_ratio:
            dst_dir = 'test/'

        dst = dataset_location + dst_dir + 'fire/' + image
        if source.endswith('no_fire/'):
            dst = dataset_location + dst_dir + 'no_fire/' + image

        copyfile(src, dst)

        counter += 1

        if counter == max_img:
            break

