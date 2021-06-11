#!/bin/bash
echo script name: $0
echo $# arguments

dataset=xray

python --version

OMP_NUM_THREADS=4 python ./exps/basic/basic-eval.py \
	--checkpoint ./output/basic/${dataset}/simple/checkpoint/seed-42-best.pth \
	--data_path $TORCH_HOME/${dataset}.python 