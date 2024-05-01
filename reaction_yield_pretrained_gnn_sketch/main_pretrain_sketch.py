import argparse
import os
import random
import numpy as np
import torch
from rdkit import Chem, rdBase

from data.get_pretraining_data import preprocess, get_mordred
from src.pretrain_sketch import pretrain

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrain_dpath", type=str, default="./data/pretraining/")
    parser.add_argument(
        "--pretrain_graph_save_path", type=str, default="../../data"
    )
    parser.add_argument(
        "--pretrain_mordred_save_path", type=str, default="../../data"
    )

    parser.add_argument("--pca_dim", type=int)
    parser.add_argument("--seed", type=int, default=27407)

    parser.add_argument('--device', type=int, default=0, choices=range(-1, 5),
                        help='GPU device index to use, -1 refers to CPU.')
    parser.add_argument('--log_steps', type=int, default=1, choices=[1, 2, 5, 10],
                        help='Number of epoches per logging.')
    parser.add_argument('--compress_ratio', type=float, default=0.8,
                        choices=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='Compression ratio for sketching modules.')
    parser.add_argument('--sampling', type=bool, default=False, choices=[True, False],
                        help='Whether to sample mini-batches.')
    parser.add_argument('--sampling_type', type=str, default='RW', choices=['RW'],
                        help='Sampling strategies.')
    parser.add_argument('--sampling_ratio', type=float, default=1.0,
                        choices=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                        help='Sampling ratio for mini-batches.')
    parser.add_argument('--num_workers', type=int, default=0, choices=range(0, 9),
                        help='Number of workers in data loaders.')
    parser.add_argument('--num_layers', type=int, default=4, choices=range(2, 6),
                        help='Number of layers of the model.')
    parser.add_argument('--hidden_channels', type=int, default=128, choices=[64, 128, 256],
                        help='Number of hidden channels.')
    parser.add_argument('--batchnorm', type=bool, default=False, choices=[True, False],
                        help='Whether to use batch normalization.')
    parser.add_argument('--dropout', type=float, default=0, choices=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help='Drop out rate.')
    parser.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'Sigmoid', 'None'],
                        help='Activation function to use.')
    parser.add_argument('--order', type=int, default=2, choices=range(1, 10),
                        help='Order of (approximated) polynomial activation.')
    parser.add_argument('--top_k', type=int, default=8, choices=range(1, 17),
                        help='Top number of entries per row to preserve in the sketched convolution matrices.')
    parser.add_argument('--mode', type=str, default='all_same', # sketching mode
                        choices=['all_distinct', 'layer_distinct', 'order_distinct', 'all_same'],
                        help='How are the different sketch modules different with respect to each others.')
    parser.add_argument('--lr', type=float, default=1e-3, choices=[1e-1, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
                        help='Learning rate.')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'],
                        help='Optimization algorithm to use.')
    parser.add_argument('--clip_threshold', type=float, default=0.01, choices=[0.01, 0.02, 0.05, 0.1],
                        help='Grad clipping threshold.')
    parser.add_argument('--num_sketches', type=int, default=2, choices=[1, 2, 3, 4, 5],
                        help='Number of sketches in each experiment.')
    parser.add_argument('--num_epochs', type=int, default=500, choices=[10, 20, 50, 100, 200, 300, 500, 1000],
                        help='Number of epochs in each experiment.')

    args = parser.parse_args()

    if not os.path.exists("./model/pretrained/"):
        os.makedirs("./model/pretrained/")

    pretrain(args)