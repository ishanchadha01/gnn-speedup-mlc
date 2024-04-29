import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from dgl.nn.pytorch import GINEConv
from dgl.nn.pytorch.glob import AvgPooling
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



class linear_head(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(linear_head, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.mlp = nn.Sequential(nn.Linear(in_feats, out_feats))

    def forward(self, x):
        return self.mlp(x)


class GIN(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        depth=3,
        node_hid_feats=300,
        readout_feats=1024,
        dr=0.1,
    ):
        super(GIN, self).__init__()

        self.depth = depth

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU()
        )

        self.project_edge_feats = nn.Sequential(
            nn.Linear(edge_in_feats, node_hid_feats)
        )

        self.gnn_layers = nn.ModuleList(
            [
                GINEConv(
                    apply_func=nn.Sequential(
                        nn.Linear(node_hid_feats, node_hid_feats),
                        nn.ReLU(),
                        nn.Linear(node_hid_feats, node_hid_feats),
                    )
                )
                for _ in range(self.depth)
            ]
        )

        self.readout = AvgPooling()

        self.sparsify = nn.Sequential(
            nn.Linear(node_hid_feats, readout_feats), nn.PReLU()
        )

        self.dropout = nn.Dropout(dr)

    def forward(self, g):
        node_feats_orig = g.ndata["attr"]
        edge_feats_orig = g.edata["edge_attr"]

        node_feats_init = self.project_node_feats(node_feats_orig)
        node_feats = node_feats_init
        edge_feats = self.project_edge_feats(edge_feats_orig)

        for i in range(self.depth):
            node_feats = self.gnn_layers[i](g, node_feats, edge_feats)

            if i < self.depth - 1:
                node_feats = nn.functional.relu(node_feats)

            node_feats = self.dropout(node_feats)

        readout = self.readout(g, node_feats)
        readout = self.sparsify(readout)

        return readout

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            own_state[name].copy_(param)
            print(f"variable {name} loaded!")

class Pretraining_Dataset:
    def __init__(self, graph_save_path, mordred_save_path, pc_num):
        self.graph_save_path = graph_save_path
        self.mordred_save_path = mordred_save_path
        self.pc_num = pc_num
        self.load()

    def load(self):
        print("mordred_pretrain.py started, pc num: ", self.pc_num)

        [mordred] = np.load(
            self.mordred_save_path + "pubchem_mordred.npz", allow_pickle=True
        )

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
            self.graph_save_path + "pubchem_graph.npz",
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

    def __getitem__(self, idx):
        g = graph(
            (
                self.src[self.e_csum[idx] : self.e_csum[idx + 1]],
                self.dst[self.e_csum[idx] : self.e_csum[idx + 1]],
            ),
            num_nodes=self.n_node[idx],
        )
        g.ndata["attr"] = torch.from_numpy(
            self.node_attr[self.n_csum[idx] : self.n_csum[idx + 1]]
        ).float()
        g.edata["edge_attr"] = torch.from_numpy(
            self.edge_attr[self.e_csum[idx] : self.e_csum[idx + 1]]
        ).float()

        n_node = self.n_node[idx].astype(int)
        mordred = self.mordred[idx].astype(float)

        return g, n_node, mordred

    def __len__(self):
        return self.n_node.shape[0]