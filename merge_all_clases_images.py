import os
import shutil
import glob
import math

src_folder = '/hdd2/srinath/dataset_augmentation_diffusers/custom_test_set_train/'
dest_folder = '/hdd2/srinath/dataset_augmentation_diffusers/ct_all_images/'

# classes = ['airplane', 'automobile',  'bird',  'cat',  'deer',  'dog',  'frog',  'horse',  'ship',  'truck']
classes = ['cat']

total_imags = 0

# get the number of files already in the dir
for cls in classes:
	img_id = 0
	final_path = os.path.join(src_folder, cls)
	print(final_path)
	for name in os.listdir(final_path):
		file_abs_path = os.path.join(final_path, name)
		if os.path.isfile(file_abs_path):
			print(f'{name = }')
			# TODO figure out automatically if the images are pngs or jpgs
			shutil.copy(file_abs_path, os.path.join(dest_folder, f'{cls}_{img_id}.jpg'))
			img_id += 1
	# num_files = len([name for name in os.listdir(final_path) if os.path.isfile(name)])


