#!/bin/bash
APP="433.milc-s0.txt.xz"
MODEL="vit.stu.90.0"
nohup taskset -c 0-245 python src/3_vit.py $APP $MODEL 16 2 &
taskset -c 0-245 python src/3_vit.py $APP $MODEL 32 2 
nohup taskset -c 0-245 python src/3_vit.py $APP $MODEL 64 2 &
taskset -c 0-245 python src/3_vit.py $APP $MODEL 128 2 
nohup taskset -c 0-245 python src/3_vit.py $APP $MODEL 256 2 &
taskset -c 0-245 python src/3_vit.py $APP $MODEL 512 2 
nohup taskset -c 0-245 python src/3_vit.py $APP $MODEL 1024 2 &
taskset -c 0-245 python src/3_vit.py $APP $MODEL 128 1 
nohup taskset -c 0-245 python src/3_vit.py $APP $MODEL 128 4 &
taskset -c 0-245 python src/3_vit.py $APP $MODEL 128 8 