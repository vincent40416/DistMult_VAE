#!/bin/bash
# 09/25 batch 1024 dim 512 lr 0.0001 
# 09/27 batch 512 dim 512 lr 0.001
batch_size=("128" "256" "512" "1024")
dim=("64" "128" "256" "512" "1024")
# may change to 0.0003
lr=("0.01" "0.005" "1e-5")
ver=("1" "2" "3")
data=("cn15k" "nl27k" "ppi5k")
epoch=2000
model=("DistMult_VAE") # ComplEx_m5_4 DistMult_VAE KEGCN_DistMult UKGE_logi_m2
GPU=0
for m in "${model[@]}";
do
for da in "${data[@]}"; 
do
        dimen=${dim[3]}
        batch_s=${batch_size[2]}
#         echo CUDA_VISIBLE_DEVICES=3 python3 -W ignore ./run/run.py --data ${da}_no_psl --models_dir train_model_d${dimen}_b${batch_s} --model $m  --batch_size ${batch_s} --no_psl --dim $dimen --save_freq 20  --epoch $epoch --lr ${lr[2]} -ver 2
        CUDA_VISIBLE_DEVICES=0 python3 -W ignore ./run/run.py --data ${da}_no_psl --models_dir train_model_d${dimen}_b${batch_s} --model $m  --batch_size ${batch_s} --no_psl --dim $dimen --save_freq 20 --semisupervised_v2 --epoch $epoch --lr ${lr[2]} -ver 3 --pre_trained True --n_neg 10 --n_hidden 256
        
#         dimen=${dim[2]}
#         CUDA_VISIBLE_DEVICES=1 python3 -W ignore ./run/run.py --data ${da}_no_psl --models_dir train_model_d${dimen}_b${batch_s} --model $m  --batch_size ${batch_s} --no_psl --dim $dimen --semisupervised_v2 --save_freq 40 --epoch $epoch --lr ${lr[2]} 
#         thefile=$(ls -d ./train_model_d512_b512_ver2/${da}_no_psl/${m}*)
#         echo ${thefile}
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