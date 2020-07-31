# GAMIT
This is the implimentation for our paper: 
Zhixiang HE, Chi-Yin Chow, Jia-Dong Zhang. GAMIT: A New Encoder-Decoder Framework with Graphical Space and Multi-grained Time for Traffic Predictions.

# Example to run:
python ../main.py --isgraph 1 --epoch 3 --sampling 0 --merge 0 --model_name GAMIT --dataset PEMS08 --len_his 6 --len_his2 6 --len_pre 6 --len_f 1 --Ks 3 --Kt 3 --num_cell 1 --filters1 64 --filters2 64 --lr 0.01 --act_func relu --decay_rate 0.5 --global_step 2000 --dr 0 --isfcl 0 --batch_size 32 --isTrain 1

# Data
Unzip data under the folder "data", which are downloaded from https://github.com/Davidham3/ASTGCN/tree/master/data