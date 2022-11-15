echo $1
rm train_images/plane/*[1-9][0-9][0-9][0-9]\.png 
rm train_images/car/*[1-9][0-9][0-9][0-9]\.png 
rm train_images/bird/*[1-9][0-9][0-9][0-9]\.png 
rm train_images/cat/*[1-9][0-9][0-9][0-9]\.png 
rm train_images/deer/*[1-9][0-9][0-9][0-9]\.png 
rm train_images/dog/*[1-9][0-9][0-9][0-9]\.png 
rm train_images/frog/*[1-9][0-9][0-9][0-9]\.png 
rm train_images/horse/*[1-9][0-9][0-9][0-9]\.png 
rm train_images/ship/*[1-9][0-9][0-9][0-9]\.png 
rm train_images/truck/*[1-9][0-9][0-9][0-9]\.png 

ls train_images/plane/ | wc -l
ls train_images/car/ | wc -l
ls train_images/bird/ | wc -l
ls train_images/cat/ | wc -l
ls train_images/deer/ | wc -l
ls train_images/dog/ | wc -l
ls train_images/frog/ | wc -l
ls train_images/horse/ | wc -l
ls train_images/ship/ | wc -l
ls train_images/truck/ | wc -l


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
