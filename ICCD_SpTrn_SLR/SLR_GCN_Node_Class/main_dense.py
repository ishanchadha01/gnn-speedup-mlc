import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
import utils 

# Parse arguments
args = utils.parse_args_dense()

# Set up random seed for Reproducibility
torch.manual_seed(args.seed)

# Set up logger file:
logger = utils.get_logger(args.logging)

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='Cora')
# parser.add_argument('--hidden_channels', type=int, default=16)
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--epochs', type=int, default=200)
# parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
# parser.add_argument('--wandb', action='store_true', help='Track experiment')
# args = parser.parse_args()

# GPU device configuration
device = torch.device("cuda:"+str(args.gpus[0]) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpus[0])
logger.info('Using device ' + ("cuda:"+str(args.gpus[0]) if torch.cuda.is_available() else "cpu") +
                     ' for training')


# track the training process
init_wandb(name=f'GCN-{args.dataset}', lr=args.lr, epochs=args.epochs,
           hidden_channels=args.hidden_channels, device=device)

# dataset downloading/configuration
dataset = Planetoid(args.dataset_dir, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0]



if args.use_gdc:
    transform = T.GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.05),
        sparsification_kwargs=dict(method='topk', k=128, dim=0),
        exact=True,
    )
    data = transform(data)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize=not args.use_gdc)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes)
model, data = model.to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=args.lr)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


logger.info('****************Start training****************')
best_val_acc = final_test_acc = 0
for epoch in range(1, args.epochs + 1):
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        torch.save(model.state_dict(), args.save_model_path)
        logger.info('Saved model to ' + args.save_model_path + ' . Validation accuracy: {:.3f}, test accuracy: {:.3f}.'.format(\
                val_acc, test_acc))
    logger.info('In the {}th epoch, the loss is: {:.3f}, training accuracy: {:.3f}, validation accuracy: {:.3f}, test accuracy: {:.3f}.'.format(\
        epoch, loss, train_acc, val_acc, test_acc))
    

logger.info('****************End training****************')
