import time
import numpy as np
import torch
import math
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
# from dgl.convert import graph
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.model import SketchGIN, linear_head
from src.sketch import preprocess_data

from torch_geometric.data.dataset import Dataset as GraphDataset
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

def pretrain(args):
    torch_dataset = Torch_Pretraining_Dataset(args.pretrain_graph_save_path, args.pretrain_mordred_save_path, args.pca_dim)
    pretraining_dataset = Sketch_Pretraining_Dataset(
        torch_dataset, args.compress_ratio, args.num_layers, args.order, args.top_k, args.mode, args.num_sketches
    )

    train_loader = DataLoader(
        dataset=pretraining_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    node_dim = pretraining_dataset.pyg_dataset[0].x.shape[1]
    # edge_dim = pretraining_dataset.edge_attr.shape[1]
    mordred_dim = pretraining_dataset.pyg_dataset[0].y.shape[0]
    print(f"Node dim {node_dim}, mordred dim {mordred_dim}")

    # g_encoder = SketchGIN(node_dim, edge_dim).to(device)
    g_encoder = SketchGIN(node_in_feats=node_dim, out_channels=1024)
    m_predictor = linear_head(in_feats=1024, out_feats=mordred_dim).to(device)

    # optimizer = Adam(g_encoder.parameters(),lr=args.lr,weight_decay=args.l2)
    mask = None

    pc_eigenvalue = pretraining_dataset.pyg_dataset.pc_eigenvalue

    pretrain_moldescpred(args, g_encoder, m_predictor, train_loader, pc_eigenvalue, args.seed, mask)


def pretrain_moldescpred(
    args,
    g_encoder,
    m_predictor,
    trn_loader,
    pc_eigenvalue,
    seed,
    mask
):
    max_epochs = 10

    pretrained_model_path = "./model/pretrained/" + "%d_pretrained_gnn.pt" % (seed)

    optimizer = Adam(
        list(g_encoder.parameters()) + list(m_predictor.parameters()),
        lr=5e-4,
        weight_decay=1e-5,
    )

    pc_eigenvalue = torch.from_numpy(pc_eigenvalue).to(device)

    def weighted_mse_loss(input, target, weight):
        return (weight * ((input - target) ** 2)).mean()

    l_start_time = time.time()

    trn_size = trn_loader.dataset.__len__()
    batch_size = trn_loader.batch_size

    

    for epoch in range(max_epochs):
        g_encoder.train()
        m_predictor.train()

        # # model.train()
        # optimizer.zero_grad()
        # out = model(data.x, data.edge_index, data.edge_weight)
        # loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        # loss.backward()
        # # optimizer.step()
        # if mask is not None: mask.step()
        # else: optimizer.step()

        start_time = time.time()

        trn_loss_list = []

        for batchidx, batchdata in tqdm(
            enumerate(trn_loader), total=trn_size // batch_size, leave=False
        ):
            # inputs, n_nodes, mordred = batchdata
            nf_sketches, ef_sketches, conv_sketches, ll_cs_list, label = batchdata
            # nf_sketches = nf_sketches.to(device)
            # ef_sketches = ef_sketches.to(device)
            # conv_sketches = conv_sketches.to(device)
            # ll_cs_list = ll_cs_list.to(device)
            label = label.to(device)

            # inputs = inputs.to(device)

            # mordred = mordred.to(device)

            # g_rep = g_encoder(nf_sketches, ef_sketches, conv_mat)
            # m_pred = m_predictor(g_rep)


            optimizer.zero_grad()

            outs = []
            for i in range(args.num_sketches):
                out_sketches = g_encoder(nf_sketches[i], ef_sketches[i], conv_sketches[i])
                outs.extend([cs.unsketch_mat(os) for cs, os in zip(ll_cs_list[i], out_sketches)])
            out = torch.median(torch.stack(outs, dim=0), dim=0).values
            # out = out[train_idx] #TODO: train/test split with train idx
            # loss = F.nll_loss(out, label.squeeze(1)[train_idx])
            out = m_predictor(out)
            out = torch.mean(out, dim=0)

            # loss = F.nll_loss(out.to(dtype=torch.float32), label.to(dtype=torch.float32))
            loss = weighted_mse_loss(out, label, pc_eigenvalue)
            loss.backward()

            optimizer.step()

            train_loss = loss.detach().item()
            trn_loss_list.append(train_loss)

        printed_train_loss = np.mean(trn_loss_list)
        print(
            "---epoch %d, lr %f, train_loss %.6f, time_per_epoch %f"
            % (
                epoch,
                optimizer.param_groups[-1]["lr"],
                printed_train_loss,
                (time.time() - start_time) / 60,
            )
        )

    torch.save(g_encoder.state_dict(), pretrained_model_path)

    print("pretraining terminated!")
    print("learning time (min):", (time.time() - l_start_time) / 60)



class Torch_Pretraining_Dataset(GraphDataset):
    def __init__(self, graph_save_path, mordred_save_path, pc_num):
        self.graph_save_path = graph_save_path
        self.mordred_save_path = mordred_save_path
        self.pc_num = pc_num
        self.load()
        super(Torch_Pretraining_Dataset, self).__init__(transform=T.ToSparseTensor())

    def load(self):
        print("mordred_pretrain.py started, pc num: ", self.pc_num)

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

        # Standardizing descriptors to have a mean of zero and std of one
        scaler = StandardScaler()
        mordred = scaler.fit_transform(mordred)

        # Applying PCA to reduce the dimensionality of descriptors
        pca = PCA(n_components=self.pc_num)
        mordred = pca.fit_transform(mordred)
        self.pc_eigenvalue = pca.explained_variance_
        print("eigenvalue:", self.pc_eigenvalue)

        # Clipping each dimension to -10*std ~ 10*std
        mordred = np.clip(mordred, -np.std(mordred, 0) * 10, np.std(mordred, 0) * 10)

        # Re-standardizing descriptors
        scaler = StandardScaler()
        mordred = scaler.fit_transform(mordred)

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

    # def __getitem__(self, idx):
    #     g = graph(
    #         (
    #             self.src[self.e_csum[idx] : self.e_csum[idx + 1]],
    #             self.dst[self.e_csum[idx] : self.e_csum[idx + 1]],
    #         ),
    #         num_nodes=self.n_node[idx],
    #     )
    #     g.ndata["attr"] = torch.from_numpy(
    #         self.node_attr[self.n_csum[idx] : self.n_csum[idx + 1]]
    #     ).float()
    #     g.edata["edge_attr"] = torch.from_numpy(
    #         self.edge_attr[self.e_csum[idx] : self.e_csum[idx + 1]]
    #     ).float()

    #     n_node = self.n_node[idx].astype(int)
    #     mordred = self.mordred[idx].astype(float)

    #     return g, n_node, mordred
    # def __getitem__(self, idx):
    #     return self.get(idx)
    
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

    def __len__(self):
        return self.n_node.shape[0]
        # return 100 # temp for debugging
    
    def len(self):
        return self.__len__()
    

class Sketch_Pretraining_Dataset(Dataset):
    def __init__(self, torch_dataset, compress_ratio, num_layers, order, top_k, mode, num_sketches):
        super(Sketch_Pretraining_Dataset, self).__init__()
        # self.pyg_dataset = torch_dataset[:100] # temp for debugging
        self.pyg_dataset = torch_dataset
        self.compress_ratio = compress_ratio
        self.num_layers = num_layers
        self.order = order
        self.top_k = top_k
        self.mode = mode
        self.num_sketches = num_sketches
        self.nf_sketches = []
        self.ef_sketches = []
        self.conv_sketches = []
        self.ll_cs_list = []
        self.labels = []
        self.__preprocess__()

    def __preprocess__(self):
        print("Creating datalists:")
        for idx in tqdm(range(len(self.pyg_dataset))):
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

            # self.data_list.append((nf_mat, ef_mat, conv_mat, label))

            self.nf_sketches.append([])
            self.ef_sketches.append([])
            self.conv_sketches.append([])
            self.ll_cs_list.append([])
            top_k = min(self.top_k, conv_mat.size(0)-4) # If too small, dont need to sparsify anyways. -4 is a arbitrary setting though
            for _ in range(self.num_sketches):
                nf_sketches, ef_sketches, conv_sketches, ll_cs_list = preprocess_data(
                    self.num_layers, in_dim=nf_mat.size(0), in_edge_dim=ef_mat.size(0),
                    out_dim=math.ceil(nf_mat.size(0) * self.compress_ratio), order=self.order, top_k=top_k,
                    mode=self.mode, nf_mat=nf_mat, ef_mat=ef_mat, conv_mat=conv_mat,
                )
                self.nf_sketches[-1].append(nf_sketches)
                self.ef_sketches[-1].append(ef_sketches)
                self.conv_sketches[-1].append(conv_sketches)
                self.ll_cs_list[-1].append(ll_cs_list)
            self.labels.append(label)

    def __getitem__(self, idx):
        return self.nf_sketches[idx], self.ef_sketches[idx], self.conv_sketches[idx], self.ll_cs_list[idx], self.labels[idx]

    def __len__(self):
        return len(self.nf_sketches)
