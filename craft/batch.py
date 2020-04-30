import os

for i in range(6):
    os.system("CUDA_VISIBLE_DEVICES=0 python test.py --trained_model=../craft_mlt_25k.pth --test_folder=../data/" + str(i) + "/jpg --result_folder=./result/"  + str(i) + "/")