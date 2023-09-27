#!/bin/bash
APP="410.bwaves-s0.txt.xz"
MODEL="vit"
nohup taskset -c 0-245 python src/3_vit.py $APP $MODEL 16 2 &
taskset -c 0-245 python src/3_vit.py $APP $MODEL 32 2 
nohup taskset -c 0-245 python src/3_vit.py $APP $MODEL 64 2 &
taskset -c 0-245 python src/3_vit.py $APP $MODEL 128 2 
nohup taskset -c 0-245 python src/3_vit.py $APP $MODEL 256 2 &
taskset -c 0-245 python src/3_vit.py $APP $MODEL 512 2 
nohup taskset -c 0-245 python src/3_vit.py $APP $MODEL 1024 2 &
taskset -c 0-245 python src/3_vit.py $APP $MODEL 128 1 
nohup taskset -c 0-245 python src/3_vit.py $APP 128 4 &
taskset -c 0-245 python src/3_vit.py $APP $MODEL 128 8 