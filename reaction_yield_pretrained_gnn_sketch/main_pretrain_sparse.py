import argparse
import os
import random
import numpy as np
import torch
from rdkit import Chem, rdBase
import os.path as osp

import time

from data.get_pretraining_data import preprocess, get_mordred
from src.pretrain_sparse import pretrain

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--pretrain_dpath", type=str, default="./data/pretraining/")
    arg_parser.add_argument(
        "--pretrain_graph_save_path", type=str, default="./data/pretraining/"
    )
    arg_parser.add_argument(
        "--pretrain_mordred_save_path", type=str, default="./data/pretraining/"
    )

    arg_parser.add_argument("--pca_dim", type=int)
    arg_parser.add_argument("--seed", type=int, default=27407)

    #sparse training args
    arg_parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    arg_parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    arg_parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    arg_parser.add_argument('--epochs', type=int, default=1024, metavar='N',
                        help='number of epochs to train (default: 100)')
    arg_parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.1)')
    arg_parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    arg_parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # arg_parser.add_argument('--seed', type=int, default=18, metavar='S', help='random seed (default: 17)')
    arg_parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    arg_parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = ''.join(str(time.time()).split('.'))
    arg_parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    # arg_parser.add_argument('--data', type=str, default='cifar10')
    arg_parser.add_argument('--decay_frequency', type=int, default=30000)
    arg_parser.add_argument('--l1', type=float, default=0.0)
    arg_parser.add_argument('--gamma', type=float, default=0.1)
    arg_parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    arg_parser.add_argument('--valid_split', type=float, default=0.1)
    arg_parser.add_argument('--resume', type=str)
    arg_parser.add_argument('--start-epoch', type=int, default=1)
    arg_parser.add_argument('--l2', type=float, default=5.0e-4)
    arg_parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    arg_parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    arg_parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    arg_parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    arg_parser.add_argument('--input_size', type=int, default=64, metavar='N',
                        help='number of epochs to train (default: 100)')
    arg_parser.add_argument('--hidden1_size', type=int, default=256, metavar='N',
                        help='number of epochs to train (default: 256)')
    arg_parser.add_argument('--hidden2_size', type=int, default=128, metavar='N',
                        help='number of epochs to train (default: 128)')
    arg_parser.add_argument('--num_classes', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    arg_parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name: Cora, Pubmed, or CiteSeer')
    arg_parser.add_argument('--hidden_channels', type=int, default=16)
    arg_parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    arg_parser.add_argument('--wandb', action='store_true', help='Track experiment')

    arg_parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                        '`all` indicates use all gpus.')            
    arg_parser.add_argument('--running-dir', type=str, default=osp.dirname(osp.realpath(__file__)), help='Running directory')

    arg_parser.add_argument('--model-name', default='Cora_dense', help='Stored model name')
    arg_parser.add_argument('--log-name', default='Cora_dense_train', help='Log file name')
    arg_parser.add_argument('--data_path', type=str)

    arg_parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')
    arg_parser.add_argument('--fix', action='store_true', help='Fix sparse connectivity during training. Default: True.')
    # arg_parser.add_argument('--sparse_init', type=str, default='GMP', help='sparse initialization')
    arg_parser.add_argument('--sparse_init', type=str, default='ERK', help='sparse initialization')
    arg_parser.add_argument('--growth', type=str, default='gradient', help='Growth mode. Choose from: momentum, random, random_unfired, gradient')
    arg_parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold.')
    arg_parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    arg_parser.add_argument('--death-rate', type=float, default=0.50, help='The pruning rate / death rate.')
    arg_parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    arg_parser.add_argument('--update_frequency', type=int, default=1000, metavar='N', help='how many iterations to train between parameter exploration')
    arg_parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    arg_parser.add_argument('--theta', type=float, default=1e-5, help='upper confidence bound coefficient')
    # arg_parser.add_argument('--theta_min', type=float, default=1e-30, help='upper confidence bound coefficient')
    arg_parser.add_argument('--theta_decay_freq', type=float, default=400, help='theta decay frequency')
    arg_parser.add_argument('--epsilon', type=float, default=1.0, help='upper confidence bound remainder')
    arg_parser.add_argument('--factor', type=float, default=1.0, help='theta linear decay factor')

    args = arg_parser.parse_args()
    args.sparse = True

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False

    



    # We use the demo dataset (10k mols) for convenience in github repo.
    # The full dataset (10M mols collected from Pubchem) can be downloaded from
    # https://arxiv.org/pdf/2010.09885.pdf
    molsuppl = Chem.SmilesMolSupplier(
        args.pretrain_dpath + "pubchem-10k.txt", delimiter=","
    )

    if not os.path.exists(args.pretrain_graph_save_path + "pubchem_graph.npz"):
        preprocess(molsuppl, args.pretrain_graph_save_path)

    if not os.path.exists(args.pretrain_mordred_save_path + "pubchem_mordred.npz"):
        get_mordred(molsuppl, args.pretrain_mordred_save_path)

    if not os.path.exists("./model/pretrained/"):
        os.makedirs("./model/pretrained/")



    pretrain(args)
