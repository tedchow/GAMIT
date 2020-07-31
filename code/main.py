"""
Created on Dec 29 2019,
By zhixiang
"""
from func import trainer, tester
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--isTrain", type=int, default=1, help="train: 1, test: 0")
parser.add_argument("--dataset", type=str, default='PEMS08', help="PEMS08, PEMS04")
parser.add_argument("--model_name", type=str, default='GAMIT', help=" model name")


parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--epoch", type=int, default=100, help="batch_size")
parser.add_argument("--num_cell", type=int, default=1, help="num cells")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--rg", type=float, default=0, help="learning rate")
parser.add_argument("--dr", type=float, default=0, help="dropout rate")
parser.add_argument("--decay_epoch", type=int, default=10, help="decay epochs")
parser.add_argument("--global_step", type=int, default=1000, help="global_step")
parser.add_argument("--decay_steps", type=int, default=2000, help="decay_steps if use sampling during training")
parser.add_argument("--decay_rate", type=float, default=0.1, help="learning rate decay rate")
parser.add_argument("--Ks", type=int, default=3, help="spatial kernel size")
parser.add_argument("--Kt", type=int, default=3, help="temporal kernel size")
parser.add_argument("--act_func", type=str, default='relu', help="gpu_id")


parser.add_argument("--len_his", type=int, default=6, help="length of history series of inputs")
parser.add_argument("--len_his2", type=int, default=6, help="length of history series of a slice in inputs")
parser.add_argument("--len_pre", type=int, default=6, help="length to predict")
parser.add_argument("--filters1", type=int, default=32, help="filters in GCN")
parser.add_argument("--filters2", type=int, default=32, help="filters in LSTM")


parser.add_argument("--gpu_id", type=str, default='0', help="gpu_id")
parser.add_argument("--base_path", type=str, default='../data/', help="base path")
parser.add_argument("--isgraph", type=int, default=1, help="use gcn or not")
parser.add_argument("--len_f", type=int, default=1, help="number of features")
parser.add_argument("--isfcl", type=int, default=0, help="number of features")
parser.add_argument("--sampling", type=int, default=1, help="number of features")
parser.add_argument("--merge", type=int, default=0, help="whether merge valid data when trainging")

args = parser.parse_args()


if __name__=='__main__':
    # print(args.model_name)
    if args.isTrain == 1:
        trainer(args)

    elif args.isTrain == 0:
        tester(args)


