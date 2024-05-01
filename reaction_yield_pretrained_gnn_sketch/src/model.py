import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from dgl.nn.pytorch import GINEConv
from dgl.nn.pytorch.glob import AvgPooling
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.util import MC_dropout

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'


class Edge2Node(nn.Module):
    """Converts ef mat to nf mat shape"""
    def __init__(self, input_dim=8, output_dim=155):
        super(Edge2Node, self).__init__()
        self.reducer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.reducer(x)


class linear_head(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(linear_head, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.mlp = nn.Sequential(nn.Linear(in_feats, out_feats))

    def forward(self, x):
        return self.mlp(x)


class SketchGINConv(nn.Module):
    def __init__(self, in_channels, out_channels, order):
        super(SketchGINConv, self).__init__()
        self.order = order
        self.weight = Parameter(torch.empty((in_channels, out_channels), dtype=torch.float32), requires_grad=True)
        self.bias = Parameter(torch.empty(out_channels, dtype=torch.float32), requires_grad=True)
        self.coeffs = Parameter(torch.empty(order, dtype=torch.float32), requires_grad=True)
        self.ef_nf_converter = Edge2Node()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        self.coeffs.data.fill_(0)
        self.coeffs.data[0] = 1.0

    def forward(self, nf_mats, ef_mats, conv_mats):
        ef_mats = [self.ef_nf_converter(ef_mat) for ef_mat in ef_mats]
        if self.training:
            zs = [torch.fft.rfft((nf_mats[0] + ef_mats[0]) @ self.weight + self.bias, dim=0)]
            for degree in range(2, self.order + 1):
                zs.append(torch.fft.rfft((nf_mats[degree - 1] + ef_mats[degree - 1]) @ self.weight + self.bias, dim=0))
                zs[-1] = zs[-1] * zs[-2]
            zs = list(map(lambda _: torch.fft.irfft(_, dim=0), zs))
            return [sum([self.coeffs[degree - 1] * (c @ z) for degree, c, z in zip(range(1, self.order + 1), cs, zs)])
                    for cs in conv_mats]
        else:
            zs = conv_mats @ ((nf_mats + ef_mats) @ self.weight + self.bias)
            return sum([self.coeffs[degree - 1] * torch.pow(zs, degree)
                        for degree in range(1, self.order + 1)])



class SketchGIN(nn.Module):
    def __init__(
        self,
        node_in_feats,
        out_channels,
        batchnorm=False,
        dropout=0,
        order=2,
        depth=3,
        # node_hid_feats=300
    ):
        super(SketchGIN, self).__init__()
        node_hid_feats = node_in_feats
        self.order = order
        self.convs = torch.nn.ModuleList()
        self.convs.append(SketchGINConv(node_in_feats, node_hid_feats, order))
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(node_hid_feats))
        for _ in range(depth - 2):
            self.convs.append(
                SketchGINConv(node_hid_feats, node_hid_feats, order))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(node_hid_feats))
        self.convs.append(SketchGINConv(node_hid_feats, out_channels, order))
        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, nf_mats, ef_mats, conv_mats):
        if self.training:
            nf_mats = self.convs[0](nf_mats, ef_mats, conv_mats[0])
            if self.batchnorm:
                nf_mats = [self.bns[0](nf_mat) for nf_mat in nf_mats]
            for i, conv in enumerate(self.convs[1:-1]):
                nf_mats_add = conv(nf_mats, ef_mats, conv_mats[i+1])
                nf_mats = [nf_mat + nf_mat_add for nf_mat, nf_mat_add in zip(nf_mats, nf_mats_add)]
                if self.batchnorm:
                    nf_mats = [self.bns[i+1](nf_mat) for nf_mat in nf_mats]
                nf_mats = [F.dropout(nf_mat, p=self.dropout, training=self.training) for nf_mat in nf_mats]
            nf_mats = self.convs[-1](nf_mats, ef_mats, conv_mats[-1])
            return nf_mats
        else:
            nf_mats = self.convs[0](nf_mats, ef_mats, conv_mats)
            if self.batchnorm:
                nf_mats = self.bns[0](nf_mats)
            for i, conv in enumerate(self.convs[1:-1]):
                nf_mats = nf_mats + conv(nf_mats, ef_mats, conv_mats)
                if self.batchnorm:
                    nf_mats = self.bns[i+1](nf_mats)
                nf_mats = F.dropout(nf_mats, p=self.dropout, training=self.training)
            nf_mats = self.convs[-1](nf_mats, ef_mats, conv_mats)
            return nf_mats


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


class reactionMPNN(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        pretrained_model_path,
        readout_feats=1024,
        predict_hidden_feats=512,
        prob_dropout=0.1,
    ):
        super(reactionMPNN, self).__init__()

        self.mpnn = GIN(node_in_feats, edge_in_feats)
        state_dict = torch.load(
            pretrained_model_path,
            map_location=torch.device(device),
        )
        self.mpnn.load_my_state_dict(state_dict)
        print("Successfully loaded pretrained model!")

        self.predict = nn.Sequential(
            nn.Linear(2 * readout_feats, predict_hidden_feats),
            nn.PReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, predict_hidden_feats),
            nn.PReLU(),
            nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, 2),
        )

    def forward(self, rmols, pmols):
        r_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in rmols]), 0)
        p_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in pmols]), 0)

        concat_feats = torch.cat([r_graph_feats, p_graph_feats], 1)
        out = self.predict(concat_feats)

        return out[:, 0], out[:, 1]


def training(
    net,
    train_loader,
    val_loader,
    train_y_mean,
    train_y_std,
    model_path,
    val_monitor_epoch=400,
    n_forward_pass=5,
    cuda=torch.device(device),
):
    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size

    try:
        rmol_max_cnt = train_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.dataset.pmol_max_cnt

    except:
        rmol_max_cnt = train_loader.dataset.rmol_max_cnt
        pmol_max_cnt = train_loader.dataset.pmol_max_cnt

    loss_fn = nn.MSELoss(reduction="none")

    n_epochs = 500
    optimizer = Adam(net.parameters(), lr=5e-4, weight_decay=1e-5)

    lr_scheduler = MultiStepLR(
        optimizer, milestones=[400, 450], gamma=0.1, verbose=False
    )

    for epoch in range(n_epochs):
        # training
        net.train()
        start_time = time.time()

        train_loss_list = []

        for batchidx, batchdata in enumerate(train_loader):
            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_pmol = [
                b.to(cuda)
                for b in batchdata[rmol_max_cnt : rmol_max_cnt + pmol_max_cnt]
            ]

            labels = (batchdata[-1] - train_y_mean) / train_y_std
            labels = labels.to(cuda)

            pred, logvar = net(inputs_rmol, inputs_pmol)

            loss = loss_fn(pred, labels)
            loss = (1 - 0.1) * loss.mean() + 0.1 * (
                loss * torch.exp(-logvar) + logvar
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.detach().item()
            train_loss_list.append(train_loss)

        if (epoch + 1) % 10 == 0:
            print(
                "--- training epoch %d, lr %f, processed %d/%d, loss %.3f, time elapsed(min) %.2f"
                % (
                    epoch,
                    optimizer.param_groups[-1]["lr"],
                    train_size,
                    train_size,
                    np.mean(train_loss_list),
                    (time.time() - start_time) / 60,
                )
            )

        lr_scheduler.step()

        # validation with test set
        if val_loader is not None and (epoch + 1) % val_monitor_epoch == 0:
            net.eval()
            MC_dropout(net)

            val_y = val_loader.dataset.dataset.yld[val_loader.dataset.indices]
            val_y_pred, _, _ = inference(
                net,
                val_loader,
                train_y_mean,
                train_y_std,
                n_forward_pass=n_forward_pass,
            )

            result = [
                mean_absolute_error(val_y, val_y_pred),
                mean_squared_error(val_y, val_y_pred) ** 0.5,
                r2_score(val_y, val_y_pred),
            ]
            print(
                "--- validation at epoch %d, processed %d, current MAE %.3f RMSE %.3f R2 %.3f"
                % (epoch, len(val_y), result[0], result[1], result[2])
            )

    print("training terminated at epoch %d" % epoch)
    torch.save(net.state_dict(), model_path)

    return net


def inference(
    net,
    test_loader,
    train_y_mean,
    train_y_std,
    n_forward_pass=30,
    cuda=torch.device("cuda:0"),
):
    batch_size = test_loader.batch_size

    try:
        rmol_max_cnt = test_loader.dataset.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.dataset.pmol_max_cnt

    except:
        rmol_max_cnt = test_loader.dataset.rmol_max_cnt
        pmol_max_cnt = test_loader.dataset.pmol_max_cnt

    net.eval()
    MC_dropout(net)

    test_y_mean = []
    test_y_var = []

    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
            inputs_rmol = [b.to(cuda) for b in batchdata[:rmol_max_cnt]]
            inputs_pmol = [
                b.to(cuda)
                for b in batchdata[rmol_max_cnt : rmol_max_cnt + pmol_max_cnt]
            ]

            mean_list = []
            var_list = []

            for _ in range(n_forward_pass):
                mean, logvar = net(inputs_rmol, inputs_pmol)
                mean_list.append(mean.cpu().numpy())
                var_list.append(np.exp(logvar.cpu().numpy()))

            test_y_mean.append(np.array(mean_list).transpose())
            test_y_var.append(np.array(var_list).transpose())

    test_y_mean = np.vstack(test_y_mean) * train_y_std + train_y_mean
    test_y_var = np.vstack(test_y_var) * train_y_std**2

    test_y_pred = np.mean(test_y_mean, 1)
    test_y_epistemic = np.var(test_y_mean, 1)
    test_y_aleatoric = np.mean(test_y_var, 1)

    return test_y_pred, test_y_epistemic, test_y_aleatoric
