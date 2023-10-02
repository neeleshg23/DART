#!/bin/bash
python src/3_2_vit_finetune.py $1 $2 $3 $4
python src/generate_amm.py $1 $2 $3 $4 0
