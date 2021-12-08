#!/bin/bash

data=("cn15k" "nl27k" "ppi5k")
dim="512"
batch_size="512"
model=("DistMult_VAE") # DistMult_VAE_v3_1020 logi_1027 UKGE_logi_m2_1026 DistMult_VAE_v2_1029
start="700"
end="721"
step="20"
ver="3"
for m in "${model[@]}";
do
for da in "${data[@]}"; 
do
        resume_model_path=train_model_d${dim}_b${batch_size}/${da}_no_psl/${m}_v${ver}_1024_1206
#         echo CUDA_VISIBLE_DEVICES=2 python3 -W ignore ./run/test.py --data ${da}_no_psl --model $m --dim ${dim} --batch_size ${batch_size} --no_psl --resume_model_path ${resume_model_path} --start $start --to $end --step ${step} -ver 3

        CUDA_VISIBLE_DEVICES=2 python3 -W ignore ./run/test.py --data ${da}_no_psl --model $m --dim ${dim} --batch_size ${batch_size} --no_psl --resume_model_path ${resume_model_path} --start $start --to $end --step ${step} -ver 3

        
#         python3 run/test.py --data ${da}_no_psl --batch_size 512 -d 512 --no_psl --resume_model_path ${thefile} -m ${m} --start 200 --to 500 --step 40
#         GPU=$(($GPU + 1))
done
done

# read -n1 -s -r -p $'Press space to continue...\n' key

# if [ "$key" = ' ' ]; then
# for m in "${model[@]}";
# do
#     for da in "${data[@]}"; 
#     do
#             python3 -W ignore run/test.py --data nl27k --batch_size 512 -d 512 --no_psl --resume_model_path train_model_d512_b512/nl27k/DistMult_VAE_1001 -m DistMult_VAE --start 1500 --to 2000 --step 10
#     done
# done
# python3 -W ignore run/test.py --data cn15k --batch_size 512 -d 512 --no_psl --resume_model_path train_model_d512_b512/cn15k/DistMult_VAE_1001 -m DistMult_VAE --start 600 --to 1000 --step 40


# python3 run/run.py --data cn15k_no_psl --batch_size 512 --epoch 3000 -d 512 --no_psl --models_dir  ./trained_model_batch512_dim_512_semisupervised_v2 -m ComplEx_m5_4  --semisupervised_v2 --save_freq 40

# python3 run/test.py --data cn15k_no_psl --batch_size 512 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512_semisupervised_v2/cn15k_no_psl/ComplEx_m5_4_0930 -m ComplEx_m5_4 --start 10 --to 3000 --step 20