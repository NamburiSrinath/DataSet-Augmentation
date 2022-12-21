conda activate SD_Aug
# classes=(airplane automobile bird cat deer dog frog horse ship truck)
classes=(dog frog horse ship truck)

for var in ${!classes[@]}
do
echo ${classes[$var]}
echo ${ckpt_paths[$var]}
python /hdd2/srinath/TTUR/fid.py custom_test_set_train/${classes[$var]} ./vanilla_SD_generated_images/${classes[$var]} | tee fid_logs/vs_per_class/${classes[$var]}.log ;
done
