#!/bin/sh

#CUDA_VISIBLE_DEVICES=1 python test1.py --value 1
# python test1.py --value 21
#CUDA_VISIBLE_DEVICES=0  python test1.py --value 51
#CUDA_VISIBLE_DEVICES=1  python test1.py --value 51


#CUDA_VISIBLE_DEVICES=0 python main2.py --dataset multi --source real --source2 clipart --target sketch
#CUDA_VISIBLE_DEVICES=0 python main2.py --dataset multi --source real --source2 clipart --target painting


#CUDA_VISIBLE_DEVICES=2 python main2.py --dataset multi --source real --source2 painting --target clipart --GAN_lambda 0.1
#CUDA_VISIBLE_DEVICES=2 python main2.py --dataset multi --source real --source2 clipart --target sketch
#CUDA_VISIBLE_DEVICES=2 python main2.py --dataset multi --source painting --source2 sketch --target clipart
#CUDA_VISIBLE_DEVICES=0 python main2.py --dataset multi --source real --source2 painting --target clipart --GAN_lambda 0.5


#CUDA_VISIBLE_DEVICES=6 python main6.py --dataset multi --source real --source2 painting --target clipart
##CUDA_VISIBLE_DEVICES=6 python main6.py --dataset multi --source real --source2 clipart --target sketch
#CUDA_VISIBLE_DEVICES=6 python main6.py --dataset multi --source painting --source2 sketch --target clipart


#CUDA_VISIBLE_DEVICES=4 python main_org_change.py  --method S+T
#CUDA_VISIBLE_DEVICES=4 python main_org_change.py  --method ENT
#CUDA_VISIBLE_DEVICES=4 python main_org_change.py  --method MME
#
#CUDA_VISIBLE_DEVICES=4 python main_org_change.py  --method S+T --source painting --target clipart --source2 sketch
#CUDA_VISIBLE_DEVICES=4 python main_org_change.py  --method ENT  --source painting --target clipart --source2 sketch
#CUDA_VISIBLE_DEVICES=4 python main_org_change.py  --method MME --source painting --target clipart --source2 sketch

#############################################
#CUDA_VISIBLE_DEVICES=2 python main_DA1.py  --method MME --source real --target clipart --num 1
#
##CUDA_VISIBLE_DEVICES=1 python main_DA1.py  --method S+T --source real --target clipart --source2 sketch
#CUDA_VISIBLE_DEVICES=2 python main_DA1.py  --method MME --source real --target painting --num 1
##CUDA_VISIBLE_DEVICES=1 python main_DA1.py  --method MME_G  --source real --target clipart --source2 sketch
#
#CUDA_VISIBLE_DEVICES=2 python main_DA1.py  --method MME --source painting --target clipart --num 1
#
#CUDA_VISIBLE_DEVICES=2 python main_DA1.py  --method MME  --source clipart --target sketch --num 1
#
#
#
#CUDA_VISIBLE_DEVICES=2 python main_DA1.py  --method MME  --source sketch --target painting --num 1
#
#CUDA_VISIBLE_DEVICES=2 python main_DA1.py  --method MME --source real --target sketch --num 1
#
#CUDA_VISIBLE_DEVICES=2 python main_DA1.py  --method MME --source painting --target real --num 1
#################################################


CUDA_VISIBLE_DEVICES=1 python main_DA1.py  --steps 15000 --method MME --dataset  office_home --source Product --target Art --num 3  --source2 Art
#CUDA_VISIBLE_DEVICES=4 python main_DA1.py  --steps 15000 --method MME_Gate --dataset  office_home --source Real --target Product --num 3  --source2 Art
