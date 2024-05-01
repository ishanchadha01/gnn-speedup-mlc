# DGL_BACKEND='pytorch'
# from dgl.convert import graph
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data.dataset import Dataset as GraphDataset
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gin_conv import GINEConv

from initialize_sketch import preprocess_data


import os
from typing import Union, List, Tuple
import yaml
from tqdm import tqdm
import math
import argparse


def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


def parse_activation(activation):
    if activation == 'ReLU':
        return F.relu
    elif activation == 'Sigmoid':
        return torch.sigmoid
    elif activation == 'None':
        return lambda x: x
    else:
        assert False


class Pretraining_Dataset(GraphDataset):
    def __init__(self, transform):
        self.graph_save_path = '../../data_pre'
        # self.mordred_save_path = mordred_save_path
        # self.pc_num = pc_num
        self._preprocess()
        super(Pretraining_Dataset, self).__init__(transform=transform)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return ['../../data_pre/pyg_data.pt'] #TODO: preprocess all data at once and save it so that we dont have to create graph on data access in get()

    def _preprocess(self):
        # print("mordred_pretrain.py started, pc num: ", self.pc_num)

        [mordred] = np.load(
            os.path.join(self.graph_save_path, "pubchem_mordred.npz"), allow_pickle=True
        ) # [n_nodes (N), 1613 (num of 2d features)]

        # Eliminating descriptors with more than 10 missing values
        missing_col_idx = np.arange(mordred.shape[1])[np.sum(np.isnan(mordred), 0) > 10]
        mordred = mordred[:, np.delete(np.arange(mordred.shape[1]), missing_col_idx)]

        assert np.sum(np.isnan(mordred) > 10) == 0

        # Eliminating descriptors with all zero values
        zero_std_col_idx = np.where(np.nanstd(mordred, axis=0) == 0)[0]
        mordred = mordred[:, np.delete(np.arange(mordred.shape[1]), zero_std_col_idx)]

        # Eliminating descriptors with inf
        inf_col_idx = np.where(np.sum(mordred == np.inf, axis=0) > 0)[0]
        mordred = mordred[:, np.delete(np.arange(mordred.shape[1]), inf_col_idx)]

        # Remove mols with missing values
        non_missing_mols_idx = np.where(np.sum(np.isnan(mordred), 1) == 0)[0]
        mordred = mordred[non_missing_mols_idx]

        # TODO: normalization and further culling with PCA
        # # Standardizing descriptors to have a mean of zero and std of one
        # scaler = StandardScaler()
        # mordred = scaler.fit_transform(mordred)

        # # Applying PCA to reduce the dimensionality of descriptors
        # pca = PCA(n_components=self.pc_num)
        # mordred = pca.fit_transform(mordred)
        # self.pc_eigenvalue = pca.explained_variance_
        # print("eigenvalue:", self.pc_eigenvalue)

        # # Clipping each dimension to -10*std ~ 10*std
        # mordred = np.clip(mordred, -np.std(mordred, 0) * 10, np.std(mordred, 0) * 10)

        # # Re-standardizing descriptors
        # scaler = StandardScaler()
        # mordred = scaler.fit_transform(mordred)

        print("mordred processed finished!")

        [mol_dict] = np.load(
            os.path.join(self.graph_save_path, "pubchem_graph.npz"),
            allow_pickle=True,
        )

        self.n_node = mol_dict["n_node"][non_missing_mols_idx]
        self.n_edge = mol_dict["n_edge"][non_missing_mols_idx]

        n_csum_tmp = np.concatenate([[0], np.cumsum(mol_dict["n_node"])])
        e_csum_tmp = np.concatenate([[0], np.cumsum(mol_dict["n_edge"])])

        self.node_attr = np.vstack(
            [
                mol_dict["node_attr"][n_csum_tmp[idx] : n_csum_tmp[idx + 1]]
                for idx in non_missing_mols_idx
            ]
        )

        self.edge_attr = np.vstack(
            [
                mol_dict["edge_attr"][e_csum_tmp[idx] : e_csum_tmp[idx + 1]]
                for idx in non_missing_mols_idx
            ]
        )
        self.src = np.hstack(
            [
                mol_dict["src"][e_csum_tmp[idx] : e_csum_tmp[idx + 1]]
                for idx in non_missing_mols_idx
            ]
        )
        self.dst = np.hstack(
            [
                mol_dict["dst"][e_csum_tmp[idx] : e_csum_tmp[idx + 1]]
                for idx in non_missing_mols_idx
            ]
        )

        self.mordred = mordred

        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])
        self.length = len(non_missing_mols_idx)

    def __len__(self):
        if hasattr(self, "length"):
            return self.length
        raise Exception("Process has not been run for pretraining dataset.")

    def len(self):
        return self.__len__()
    
    def get(self, idx):
        node_feats = torch.from_numpy(
            self.node_attr[self.n_csum[idx] : self.n_csum[idx + 1]]
        ).float()
        edge_idx = torch.vstack([
            torch.from_numpy(self.src[self.e_csum[idx] : self.e_csum[idx + 1]]),
            torch.from_numpy(self.dst[self.e_csum[idx] : self.e_csum[idx + 1]])
        ])
        edge_feats = torch.from_numpy(
            self.edge_attr[self.e_csum[idx] : self.e_csum[idx + 1]]
        ).float()
        mordred = torch.from_numpy(self.mordred[idx].astype(float))
        graph = Data(x=node_feats, edge_index=edge_idx, edge_attr=edge_feats, y=mordred)
        return graph


class FullGraphDataset(Dataset):
    def __init__(self, pyg_dataset):
        super(FullGraphDataset, self).__init__()
        self.pyg_dataset = pyg_dataset
        self.data_list = []
        self.nf_mat = None
        self.conv_mat = None
        self.label = None
        self.train_idx = None
        self.graph_list = []
        self.__preprocess__()
        self.num_classes = self.pyg_dataset[0].y # mordred len for output channels

    def __preprocess__(self):
        print("Creating datalists:")
        for idx in tqdm(range(len(self.pyg_dataset))):
            # nf_mat = torch.from_numpy(
            #     self.pyg_dataset.node_attr[self.pyg_dataset.n_csum[idx] : self.pyg_dataset.n_csum[idx + 1]]
            # ).float()
            # edge_idx = torch.vstack([
            #     torch.from_numpy(self.pyg_dataset.src[self.pyg_dataset.e_csum[idx] : self.pyg_dataset.e_csum[idx + 1]]),
            #     torch.from_numpy(self.pyg_dataset.dst[self.pyg_dataset.e_csum[idx] : self.pyg_dataset.e_csum[idx + 1]])
            # ])
            # ef_mat = torch.from_numpy(
            #     self.pyg_dataset.edge_attr[self.pyg_dataset.e_csum[idx] : self.pyg_dataset.e_csum[idx + 1]]
            # ).float()
            # label = self.pyg_dataset.mordred[idx].astype(float)
            # graph = Data(x=nf_mat, edge_index=edge_idx, edge_attr=ef_mat, y=label)
            # self.graph_list.append(graph)
            nf_mat = self.pyg_dataset[idx].x.clone()
            ef_mat = self.pyg_dataset[idx].edge_attr.clone()
            conv_mat = self.pyg_dataset[idx].adj_t.to_symmetric()
            label = self.pyg_dataset[idx].y.clone()
            print(label.shape, torch.max(label), torch,min(label))

            # Normalize the adjacency matrix
            conv_mat = gcn_norm(conv_mat, edge_weight=None,
                                num_nodes=nf_mat.size(0), improved=False,
                                add_self_loops=True, dtype=torch.float32)

            # Normalize features
            nf_mat = F.batch_norm(nf_mat, running_mean=None, running_var=None, training=True)
            ef_mat = F.batch_norm(ef_mat, running_mean=None, running_var=None, training=True)

            self.data_list.append((nf_mat, ef_mat, conv_mat, label))

    def __len__(self):
        return len(self.pyg_dataset)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def get(self, idx):
        node_feats = torch.from_numpy(
            self.pyg_dataset.node_attr[self.pyg_dataset.n_csum[idx] : self.pyg_dataset.n_csum[idx + 1]]
        ).float()
        edge_idx = torch.vstack([
            torch.from_numpy(self.pyg_dataset.src[self.pyg_dataset.e_csum[idx] : self.pyg_dataset.e_csum[idx + 1]]),
            torch.from_numpy(self.pyg_dataset.dst[self.pyg_dataset.e_csum[idx] : self.pyg_dataset.e_csum[idx + 1]])
        ])
        edge_feats = torch.from_numpy(
            self.pyg_dataset.edge_attr[self.pyg_dataset.e_csum[idx] : self.pyg_dataset.e_csum[idx + 1]]
        ).float()
        mordred = self.pyg_dataset.mordred[idx].astype(float)
        graph = Data(x=node_feats, edge_index=edge_idx, edge_attr=edge_feats, y=mordred)
        return graph


def get_baseline_loader(pyg_dataset):
    dataset = FullGraphDataset(pyg_dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader


def get_dataloader(pyg_dataset, compress_ratio, sampling, sampling_type, sampling_ratio, num_layers, order, top_k, mode, num_sketches, num_workers):
    dataset = SketchFullGraphDataset(pyg_dataset, compress_ratio, num_layers, order, top_k, mode, num_sketches)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader


class SketchFullGraphDataset(Dataset):
    def __init__(self, pyg_dataset, compress_ratio, num_layers, order, top_k, mode, num_sketches):
        super(SketchFullGraphDataset, self).__init__()
        self.pyg_dataset = pyg_dataset
        self.compress_ratio = compress_ratio
        self.num_layers = num_layers
        self.order = order
        self.top_k = top_k
        self.mode = mode
        self.num_sketches = num_sketches
        # self.nf_mat = None
        # self.conv_mat = None
        # self.label = None
        # self.train_idx = None
        self.nf_sketches = []
        self.ef_sketches = []
        self.conv_sketches = []
        self.ll_cs_list = []
        self.data_list = []
        self.graph_list = []
        self.__preprocess__()

    def __preprocess__(self):
        print("Creating datalists:")
        for idx in tqdm(range(len(self.pyg_dataset))):
            # nf_mat = torch.from_numpy(
            #     self.pyg_dataset.node_attr[self.pyg_dataset.n_csum[idx] : self.pyg_dataset.n_csum[idx + 1]]
            # ).float()
            # edge_idx = torch.vstack([
            #     torch.from_numpy(self.pyg_dataset.src[self.pyg_dataset.e_csum[idx] : self.pyg_dataset.e_csum[idx + 1]]),
            #     torch.from_numpy(self.pyg_dataset.dst[self.pyg_dataset.e_csum[idx] : self.pyg_dataset.e_csum[idx + 1]])
            # ])
            # ef_mat = torch.from_numpy(
            #     self.pyg_dataset.edge_attr[self.pyg_dataset.e_csum[idx] : self.pyg_dataset.e_csum[idx + 1]]
            # ).float()
            # label = self.pyg_dataset.mordred[idx].astype(float)
            # graph = Data(x=nf_mat, edge_index=edge_idx, edge_attr=ef_mat, y=label)
            # self.graph_list.append(graph)

            # # conv_mat = to_dense_adj(edge_idx)[0]
            # # # conv_mat = edge_idx
            # conv_mat = edge_idx
            nf_mat = self.pyg_dataset[idx].x.clone()
            ef_mat = self.pyg_dataset[idx].edge_attr.clone()
            conv_mat = self.pyg_dataset[idx].adj_t.to_symmetric()
            label = self.pyg_dataset[idx].y.clone()

            # Normalize the adjacency matrix
            conv_mat = gcn_norm(conv_mat, edge_weight=None,
                                num_nodes=nf_mat.size(0), improved=False,
                                add_self_loops=True, dtype=torch.float32)

            # Normalize features
            nf_mat = F.batch_norm(nf_mat, running_mean=None, running_var=None, training=True)
            ef_mat = F.batch_norm(ef_mat, running_mean=None, running_var=None, training=True)

            # Normalize the adjacency matrix
            conv_mat = gcn_norm(conv_mat, edge_weight=None,
                                num_nodes=nf_mat.size(0), improved=False,
                                add_self_loops=True, dtype=torch.float32)

            # Normalize features
            nf_mat = F.batch_norm(nf_mat, running_mean=None, running_var=None, training=True)
            ef_mat = F.batch_norm(ef_mat, running_mean=None, running_var=None, training=True)

            self.data_list.append((nf_mat, ef_mat, conv_mat, label))

            self.nf_sketches.append([])
            self.ef_sketches.append([])
            self.conv_sketches.append([])
            self.ll_cs_list.append([])
            top_k = min(self.top_k, conv_mat.size(0)-4) # If too small, dont need to sparsify anyways. -4 is a arbitrary setting though
            for _ in range(self.num_sketches):
                nf_sketches, conv_sketches, ll_cs_list = preprocess_data(
                    self.num_layers, in_dim=nf_mat.size(0),
                    out_dim=math.ceil(nf_mat.size(0) * self.compress_ratio), order=self.order, top_k=top_k,
                    mode=self.mode, nf_mat=nf_mat, conv_mat=conv_mat,
                )
                self.nf_sketches[-1].append(nf_sketches)
                self.conv_sketches[-1].append(conv_sketches)
                self.ll_cs_list[-1].append(ll_cs_list)

    def __getitem__(self, item):
        assert item == 0
        return self.nf_sketches, self.conv_sketches, self.ll_cs_list, self.label, self.train_idx

    def __len__(self):
        return 1


class StandardGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 batchnorm, dropout, activation):
        super(StandardGCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=True, bias=True))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=True, bias=True))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=True, bias=True))
        self.dropout = dropout
        self.activation = parse_activation(activation)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, nf_mat, conv_mat):
        for i, conv in enumerate(self.convs[:-1]):
            nf_mat = conv(nf_mat, conv_mat)
            if self.batchnorm:
                nf_mat = self.bns[i](nf_mat)
            nf_mat = self.activation(nf_mat)
            nf_mat = F.dropout(nf_mat, p=self.dropout, training=self.training)
        nf_mat = self.convs[-1](nf_mat, conv_mat)
        # return nf_mat
        # return F.log_softmax(nf_mat, dim=1)
        return torch.sum(nf_mat, dim=0) # sum along node dim so that we just have len(mordred) output


class SketchGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, order):
        super(SketchGCNConv, self).__init__()
        self.order = order
        self.weight = Parameter(torch.empty((in_channels, out_channels), dtype=torch.float32), requires_grad=True)
        self.bias = Parameter(torch.empty(out_channels, dtype=torch.float32), requires_grad=True)
        self.coeffs = Parameter(torch.empty(order, dtype=torch.float32), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        self.coeffs.data.fill_(0)
        self.coeffs.data[0] = 1.0

    def forward(self, nf_mats, conv_mats):
        if self.training:
            zs = [torch.fft.rfft(nf_mats[0] @ self.weight + self.bias, dim=0)]
            for degree in range(2, self.order + 1):
                zs.append(torch.fft.rfft(nf_mats[degree - 1] @ self.weight + self.bias, dim=0))
                zs[-1] = zs[-1] * zs[-2]
            zs = list(map(lambda _: torch.fft.irfft(_, dim=0), zs))
            return [sum([self.coeffs[degree - 1] * (c @ z) for degree, c, z in zip(range(1, self.order + 1), cs, zs)])
                    for cs in conv_mats]
        else:
            zs = conv_mats @ (nf_mats @ self.weight + self.bias)
            return sum([self.coeffs[degree - 1] * torch.pow(zs, degree)
                        for degree in range(1, self.order + 1)])


class SketchGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 batchnorm, dropout, order):
        super(SketchGCN, self).__init__()
        self.order = order
        self.convs = torch.nn.ModuleList()
        self.convs.append(SketchGCNConv(in_channels, hidden_channels, order))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SketchGCNConv(hidden_channels, hidden_channels, order))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SketchGCNConv(hidden_channels, out_channels, order))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, nf_mats, conv_mats):
        if self.training:
            nf_mats = self.convs[0](nf_mats, conv_mats[0])
            if self.batchnorm:
                nf_mats = [self.bns[0](nf_mat) for nf_mat in nf_mats]
            for i, conv in enumerate(self.convs[1:-1]):
                nf_mats_add = conv(nf_mats, conv_mats[i+1])
                nf_mats = [nf_mat + nf_mat_add for nf_mat, nf_mat_add in zip(nf_mats, nf_mats_add)]
                if self.batchnorm:
                    nf_mats = [self.bns[i+1](nf_mat) for nf_mat in nf_mats]
                nf_mats = [F.dropout(nf_mat, p=self.dropout, training=self.training) for nf_mat in nf_mats]
            nf_mats = self.convs[-1](nf_mats, conv_mats[-1])
            return nf_mats
        else:
            nf_mats = self.convs[0](nf_mats, conv_mats)
            if self.batchnorm:
                nf_mats = self.bns[0](nf_mats)
            for i, conv in enumerate(self.convs[1:-1]):
                nf_mats = nf_mats + conv(nf_mats, conv_mats)
                if self.batchnorm:
                    nf_mats = self.bns[i+1](nf_mats)
                nf_mats = F.dropout(nf_mats, p=self.dropout, training=self.training)
            nf_mats = self.convs[-1](nf_mats, conv_mats)
            return nf_mats


#TODO import gineconv for 2nd baseline


def main():

    # TODO: test transform to sparse tensor
    # # load dataset
    # dataset = PygNodePropPredDataset(name='ogbn-{}'.format(args.dataset),
    #                                  transform=T.ToSparseTensor(), root="../dataset")
    # num_features = dataset[0].num_features
    # config_fp = "../../config/config_sketching.yaml"
    # with open(config_fp, 'r') as config:
    #     args = yaml.safe_load(config)

    parser = argparse.ArgumentParser()
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
    parser.add_argument('--sketch_mode', type=str, default='all_same',
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
    parser.add_argument('--runs', type=int, default=3, choices=range(1, 11),
                        help='Number of repeated experiments.')
    args = parser.parse_args()

    dataset = Pretraining_Dataset(transform=T.ToSparseTensor())
    num_features = dataset[0].num_features

    # dataloader


    train_loader1 = get_baseline_loader(dataset)
    train_loader2 = get_dataloader(dataset, args.compress_ratio, args.sampling, args.sampling_type,
                                      args.sampling_ratio, args.num_layers, args.order, args.top_k,
                                      args.sketch_mode, args.num_sketches, args.num_workers)
    
    # load models
    model2 = SketchGCN(num_features, args.hidden_channels, dataset.num_classes,
                          args.num_layers, args.batchnorm, args.dropout, args.order)
    model1 = StandardGCN(num_features, args.hidden_channels, dataset.num_classes,
                            args.num_layers, args.batchnorm, args.dropout, args.activation)
    

    # # evaluator
    # evaluator = Evaluator(name='ogbn-{}'.format(args.dataset))

    # # logger
    # logger = Logger(args.runs, args)

    # # optimizer
    # model.reset_parameters()
    # if args.optimizer == 'SGD':
    #     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # elif args.optimizer == 'Adam':
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # else:
    #     raise NotImplementedError

    # # load model
    # torch.cuda.empty_cache()
    # assert torch.cuda.memory_allocated(device) == 0
    # model = model.to(device)
    # model_memory_usage = torch.cuda.memory_allocated(device)

    # # train entrance
    # for run in range(args.runs):
    #     # reset model
    #     model.reset_parameters()

    #     # load test data
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     test_data = list(test_loader)[0]
    #     test_data = load_nested_list(test_data, device)
    #     test_memory_usage = torch.cuda.memory_allocated(device) - model_memory_usage

    #     # train loop
    #     for epoch in range(1, 1 + args.num_epochs):
    #         # initialize
    #         load_epoch_time = 0
    #         train_epoch_time = 0
    #         test_epoch_time = 0
    #         train_memory_usage = None
    #         loss = None
    #         result = None

    #         # train loop
    #         start_time = time.time()
    #         for epoch_data in train_loader:
    #             # load
    #             epoch_data = load_nested_list(epoch_data, device)
    #             load_epoch_time += time.time() - start_time

    #             # train
    #             start_time = time.time()
    #             gc.collect()
    #             torch.cuda.empty_cache()
    #             if not run_baselines:
    #                 loss = train(model, epoch_data, optimizer, args.clip_threshold, args.num_sketches)
    #             else:
    #                 loss = train_baselines(model, epoch_data, optimizer)
    #             train_memory_usage = torch.cuda.memory_allocated(device) - test_memory_usage - model_memory_usage
    #             train_epoch_time = time.time() - start_time

    #             # test
    #             start_time = time.time()
    #             if not run_baselines:
    #                 result = test(model, test_data, evaluator)
    #             else:
    #                 result = test_baselines(model, test_data, evaluator)
    #             test_epoch_time = time.time() - start_time

    #             # load
    #             start_time = time.time()

    #         # test
    #         if epoch % args.log_steps == 0:
    #             # log
    #             logger.add_result(run, result, (load_epoch_time, train_epoch_time, test_epoch_time),
    #                               (test_memory_usage, train_memory_usage))
    #             train_acc, valid_acc, test_acc = result
    #             print(f'Run: {run + 1:02d}, '
    #                   f'Epoch: {epoch:04d}, '
    #                   f'Loss: {loss:012.3f}, '
    #                   f'Train: {100 * train_acc:05.2f}%, '
    #                   f'Valid: {100 * valid_acc:05.2f}%, '
    #                   f'Test: {100 * test_acc:05.2f}%, '
    #                   f'Load Time: {load_epoch_time:06.4f}s, '
    #                   f'Train Time: {train_epoch_time:06.4f}s, '
    #                   f'Test Time: {test_epoch_time:06.4f}s, '
    #                   f'Test Memory: {test_memory_usage / 1048576.0:07.2f}MB, '
    #                   f'Train Memory: {train_memory_usage / 1048576.0:07.2f}MB')
    #         # clear
    #         del epoch_data
    #         gc.collect()
    #         torch.cuda.empty_cache()

    #     # log
    #     logger.print_statistics(run)
    # logger.print_statistics()


main()