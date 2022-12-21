# echo $1
# rm train_images/plane/*[1-9][0-9][0-9][0-9]\.png 
# rm train_images/car/*[1-9][0-9][0-9][0-9]\.png 
# rm train_images/bird/*[1-9][0-9][0-9][0-9]\.png 
# rm train_images/cat/*[1-9][0-9][0-9][0-9]\.png 
# rm train_images/deer/*[1-9][0-9][0-9][0-9]\.png 
# rm train_images/dog/*[1-9][0-9][0-9][0-9]\.png 
# rm train_images/frog/*[1-9][0-9][0-9][0-9]\.png 
# rm train_images/horse/*[1-9][0-9][0-9][0-9]\.png 
# rm train_images/ship/*[1-9][0-9][0-9][0-9]\.png 
# rm train_images/truck/*[1-9][0-9][0-9][0-9]\.png 

# echo $1
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/airplane/*[1-9][0-9][0-9][0-9]\.png 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/automobile/*[1-9][0-9][0-9][0-9]\.png 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/bird/*[1-9][0-9][0-9][0-9]\.png 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/cat/*[1-9][0-9][0-9][0-9]\.png 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/deer/*[1-9][0-9][0-9][0-9]\.png 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/dog/*[1-9][0-9][0-9][0-9]\.png 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/frog/*[1-9][0-9][0-9][0-9]\.png 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/horse/*[1-9][0-9][0-9][0-9]\.png 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/ship/*[1-9][0-9][0-9][0-9]\.png 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/truck/*[1-9][0-9][0-9][0-9]\.png 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/airplane/*[1-9][0-9][0-9][0-9]\.jpg
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/automobile/*[1-9][0-9][0-9][0-9]\.jpg 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/bird/*[1-9][0-9][0-9][0-9]\.jpg 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/cat/*[1-9][0-9][0-9][0-9]\.jpg 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/deer/*[1-9][0-9][0-9][0-9]\.jpg 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/dog/*[1-9][0-9][0-9][0-9]\.jpg 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/frog/*[1-9][0-9][0-9][0-9]\.jpg 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/horse/*[1-9][0-9][0-9][0-9]\.jpg 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/ship/*[1-9][0-9][0-9][0-9]\.jpg 
# rm /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/truck/*[1-9][0-9][0-9][0-9]\.jpg 

# ls /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/plane/ | wc -l
# ls /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/car/ | wc -l
# ls /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/bird/ | wc -l
# ls /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/cat/ | wc -l
# ls /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/deer/ | wc -l
# ls /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/dog/ | wc -l
# ls /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/frog/ | wc -l
# ls /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/horse/ | wc -l
# ls /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/ship/ | wc -l
# ls /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/truck/ | wc -l

# mkdir dreambooth_generated_images
# cp -R /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/airplane/samples/ dreambooth_generated_images/airplane
# cp -R /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/automobile/samples/ dreambooth_generated_images/automobile
# cp -R /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/bird/samples/ dreambooth_generated_images/bird
# cp -R /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/cat/samples/ dreambooth_generated_images/cat
# cp -R /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/deer/samples/ dreambooth_generated_images/deer
# cp -R /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/dog/samples/ dreambooth_generated_images/dog
# cp -R /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/frog/samples/ dreambooth_generated_images/frog
# cp -R /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/horse/samples/ dreambooth_generated_images/horse
# cp -R /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/ship/samples/ dreambooth_generated_images/ship
# cp -R /hdd2/srinath/Dreambooth-Stable-Diffusion/generated_images_5train_copy/truck/samples/ dreambooth_generated_images/truck

# mkdir text_inv_generated_images
# cp -R /hdd2/srinath/textual_inversion/text_inv_generated_images/airplane/samples/ text_inv_generated_images/airplane
# cp -R /hdd2/srinath/textual_inversion/text_inv_generated_images/automobile/samples/ text_inv_generated_images/automobile
# cp -R /hdd2/srinath/textual_inversion/text_inv_generated_images/bird/samples/ text_inv_generated_images/bird
# cp -R /hdd2/srinath/textual_inversion/text_inv_generated_images/cat/samples/ text_inv_generated_images/cat
# cp -R /hdd2/srinath/textual_inversion/text_inv_generated_images/deer/samples/ text_inv_generated_images/deer
# cp -R /hdd2/srinath/textual_inversion/text_inv_generated_images/dog/samples/ text_inv_generated_images/dog
# cp -R /hdd2/srinath/textual_inversion/text_inv_generated_images/frog/samples/ text_inv_generated_images/frog
# cp -R /hdd2/srinath/textual_inversion/text_inv_generated_images/horse/samples/ text_inv_generated_images/horse
# cp -R /hdd2/srinath/textual_inversion/text_inv_generated_images/ship/samples/ text_inv_generated_images/ship
# cp -R /hdd2/srinath/textual_inversion/text_inv_generated_images/truck/samples/ text_inv_generated_images/truck



ls text_inv_generated_images/airplane/ | wc -l
ls text_inv_generated_images/automobile/ | wc -l
ls text_inv_generated_images/bird/ | wc -l
ls text_inv_generated_images/cat/ | wc -l
ls text_inv_generated_images/deer/ | wc -l
ls text_inv_generated_images/dog/ | wc -l
ls text_inv_generated_images/frog/ | wc -l
ls text_inv_generated_images/horse/ | wc -l
ls text_inv_generated_images/ship/ | wc -l
ls text_inv_generated_images/truck/ | wc -l


ls vanilla_SD_generated_images/airplane/ | wc -l
ls vanilla_SD_generated_images/automobile/ | wc -l
ls vanilla_SD_generated_images/bird/ | wc -l
ls vanilla_SD_generated_images/cat/ | wc -l
ls vanilla_SD_generated_images/deer/ | wc -l
ls vanilla_SD_generated_images/dog/ | wc -l
ls vanilla_SD_generated_images/frog/ | wc -l
ls vanilla_SD_generated_images/horse/ | wc -l
ls vanilla_SD_generated_images/ship/ | wc -l
ls vanilla_SD_generated_images/truck/ | wc -l

ls dreambooth_generated_images/airplane | wc -l
ls dreambooth_generated_images/automobile | wc -l
ls dreambooth_generated_images/bird | wc -l
ls dreambooth_generated_images/cat/ | wc -l
ls dreambooth_generated_images/deer/ | wc -l
ls dreambooth_generated_images/dog/ | wc -l
ls dreambooth_generated_images/frog/ | wc -l
ls dreambooth_generated_images/horse/ | wc -l
ls dreambooth_generated_images/ship/ | wc -l
ls dreambooth_generated_images/truck/ | wc -l

ls custom_test_set_train/airplane | wc -l
ls custom_test_set_train/automobile | wc -l
ls custom_test_set_train/bird | wc -l
ls custom_test_set_train/cat/ | wc -l
ls custom_test_set_train/deer/ | wc -l
ls custom_test_set_train/dog/ | wc -l
ls custom_test_set_train/frog/ | wc -l
ls custom_test_set_train/horse/ | wc -l
ls custom_test_set_train/ship/ | wc -l
ls custom_test_set_train/truck/ | wc -l

ls custom_test_set_testing/airplane | wc -l
ls custom_test_set_testing/automobile | wc -l
ls custom_test_set_testing/bird | wc -l
ls custom_test_set_testing/cat/ | wc -l
ls custom_test_set_testing/deer/ | wc -l
ls custom_test_set_testing/dog/ | wc -l
ls custom_test_set_testing/frog/ | wc -l
ls custom_test_set_testing/horse/ | wc -l
ls custom_test_set_testing/ship/ | wc -l
ls custom_test_set_testing/truck/ | wc -l





# cp images_gen/plane/*["$1"][0-9][0-9][0-9]\.png train_images/plane/
# cp images_gen/car/*["$1"][0-9][0-9][0-9]\.png train_images/car/
# cp images_gen/bird/*["$1"][0-9][0-9][0-9]\.png train_images/bird/
# cp images_gen/cat/*["$1"][0-9][0-9][0-9]\.png train_images/cat/
# cp images_gen/deer/*["$1"][0-9][0-9][0-9]\.png train_images/deer/
# cp images_gen/dog/*["$1"][0-9][0-9][0-9]\.png train_images/dog/
# cp images_gen/frog/*["$1"][0-9][0-9][0-9]\.png train_images/frog/
# cp images_gen/horse/*["$1"][0-9][0-9][0-9]\.png train_images/horse/
# cp images_gen/ship/*["$1"][0-9][0-9][0-9]\.png train_images/ship/
# cp images_gen/truck/*["$1"][0-9][0-9][0-9]\.png train_images/truck/
