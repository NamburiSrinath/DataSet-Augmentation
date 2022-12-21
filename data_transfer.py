import os
import shutil
import glob
import random
import math

test_folder = '/hdd2/srinath/dataset_augmentation_diffusers/text_inv_generated_images/'
dest_folder = '/hdd2/srinath/dataset_augmentation_diffusers/text_inv_generated_images_random/'

# for folder in os.listdir(test_folder):
#     if folder[0] != '.':
#         class_folder = test_folder + folder
#         class_length = len([name for name in os.listdir(class_folder)])
#         test_set = math.ceil(0.3 * class_length)
#         to_be_moved = random.sample(os.listdir(class_folder), test_set)
#         print(len(to_be_moved))
#         for f in enumerate(to_be_moved, 1):
#             dest_class_folder = dest_folder + folder
#             if not os.path.exists(dest_class_folder):
#                 os.makedirs(dest_class_folder)
#             shutil.move(class_folder + '/' + f[1], dest_class_folder)

print("----AUGMENT DATA----")
for folder in os.listdir(test_folder):
    if folder[0] != '.':
        class_folder = test_folder + folder
        class_length = len([name for name in os.listdir(class_folder)])
        print(class_length)
print("----------")
for folder in os.listdir(dest_folder):
    if folder[0] != '.':
        class_folder = dest_folder + folder
        class_length = len([name for name in os.listdir(class_folder)])
        print(class_length)

# test_folder = '/hdd2/srinath/dataset_augmentation_diffusers/custom_test_set_copy/'
# dest_folder = '/hdd2/srinath/dataset_augmentation_diffusers/custom_test_set_testing/'

# print("---CUSTOM DATA TRAINING-----")
# for folder in os.listdir(test_folder):
#     if folder[0] != '.':
#         class_folder = test_folder + folder
#         class_length = len([name for name in os.listdir(class_folder)])
#         print(class_length)
# print("---CUSTOM DATA TEST-------")
# for folder in os.listdir(dest_folder):
#     if folder[0] != '.':
#         class_folder = dest_folder + folder
#         class_length = len([name for name in os.listdir(class_folder)])
#         print(class_length)


