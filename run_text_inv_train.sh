echo "$@"
for var in "$@"
do
echo "$var"
python main.py --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
               -t \
               --actual_resume models/ldm/text2img-large/model.ckpt \
               --placeholder_string '*' \
               -n "$var"_run_1 \
               --gpus 0,  \
               --data_root custom_train_data/$var \
               --init_word $var | tee  ./logs/custom_dataset_logs/"$var"_log.log 2>&1 ;
done
