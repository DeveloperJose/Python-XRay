#!/bin/bash
echo script name: $0
echo $# arguments

dataset=xray
batch=128
rseed=42


save_dir=./output/basic/${dataset}/simple

python --version

OMP_NUM_THREADS=4 python ./exps/basic/basic-main.py --dataset ${dataset} \
	--data_path $TORCH_HOME/${dataset}.python \
	--model_config ./configs/archs/XRAY-Arch.config \
	--optim_config ./configs/opts/XRAY-Opts.config \
	--procedure    basic \
	--save_dir     ${save_dir} \
	--cutout_length -1 \
	--batch_size  ${batch} --rand_seed ${rseed} --workers 4 \
	--eval_frequency 1 --print_freq 100 --print_freq_eval 200
