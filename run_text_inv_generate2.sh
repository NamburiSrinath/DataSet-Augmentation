classes=()
ckpt_paths=(
	/hdd2/srinath/textual_inversion/logs/dog2022-12-11T02-31-57_dog_run_1/checkpoints/embeddings_gs-6099.pt\
	/hdd2/srinath/textual_inversion/logs/frog2022-12-11T02-58-10_frog_run_1/checkpoints/embeddings_gs-6099.pt \
	/hdd2/srinath/textual_inversion/logs/horse2022-12-11T03-24-19_horse_run_1/checkpoints/embeddings_gs-6099.pt \
	/hdd2/srinath/textual_inversion/logs/ship2022-12-11T03-50-27_ship_run_1/checkpoints/embeddings_gs-6099.pt \
	/hdd2/srinath/textual_inversion/logs/truck2022-12-11T04-16-37_truck_run_1/checkpoints/embeddings_gs-6099.pt \
)

for var in ${!classes[@]}
do
echo ${classes[$var]}
echo ${ckpt_paths[$var]}
python ./txt2img.py --ddim_eta 0.0 \
                          --n_samples 1000 \
                          --n_iter 1 \
                          --scale 10.0 \ 
                          --ddim_steps 50 \ 
                          --embedding_path ${ckpt_paths[$var]} \
                          --ckpt_path  /hdd2/srinath/textual_inversion/models/ldm/text2img-large/model.ckpt \
                          --prompt "a photo of * ${classes[$var]}" \
                          --outdir text_inv_generated_images/${classes[$var]}
done


#     logs/bird2022-11-05T14-34-00_bird_run_1/checkpoints/last.ckpt \
#     logs/cat2022-11-08T00-57-31_cat_run_1/checkpoints/last.ckpt \
#     logs/deer2022-11-08T01-15-10_deer_run_1/checkpoints/last.ckpt \
#     logs/dog2022-11-08T01-33-05_dog_run_1/checkpoints/last.ckpt \
#     logs/frog2022-11-08T01-56-17_frog_run_1/checkpoints/last.ckpt \
#     logs/horse2022-11-05T12-01-01_horse_run_1/checkpoints/last.ckpt \
#     logs/plane2022-11-05T12-17-43_plane_run_1/checkpoints/last.ckpt \
#     logs/ship2022-11-05T12-34-20_ship_run_1/checkpoints/last.ckpt \
#     logs/truck2022-11-05T12-50-53_truck_run_1/checkpoints/last.ckpt

