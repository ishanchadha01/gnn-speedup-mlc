from torch.utils.data import Dataset
import numpy as np
DGL_BACKEND='pytorch'
from dgl.convert import graph
import torch

import os


class Pretraining_Dataset(Dataset):
    def __init__(self):
        self.graph_save_path = '../../data_pre'
        # self.mordred_save_path = mordred_save_path
        # self.pc_num = pc_num
        self.load()

    def load(self):
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





def main():
    p = Pretraining_Dataset()
    p3 = p[3]

    # TODO: test transform to sparse tensor
    # # load dataset
    # dataset = PygNodePropPredDataset(name='ogbn-{}'.format(args.dataset),
    #                                  transform=T.ToSparseTensor(), root="../dataset")
    # num_features = dataset[0].num_features

    # # dataloader
    # if run_baselines:
    #     train_loader = get_baseline_dataloader(dataset, args.sampling, args.sampling_type,
    #                                            args.sampling_ratio, args.num_layers, args.num_workers)
    #     test_loader = get_baseline_test_dataloader(dataset)
    # else:
    #     train_loader = get_dataloader(dataset, args.compress_ratio, args.sampling, args.sampling_type,
    #                                   args.sampling_ratio, args.num_layers, args.order, args.top_k,
    #                                   args.sketch_mode, args.num_sketches, args.num_workers)
    #     test_loader = get_test_dataloader(dataset)

    # # load model
    # if args.model == 'Sketch-GCN':
    #     model = SketchGCN(num_features, args.hidden_channels, dataset.num_classes,
    #                       args.num_layers, args.batchnorm, args.dropout, args.order)
    # elif args.model == 'Standard-GCN':
    #     model = StandardGCN(num_features, args.hidden_channels, dataset.num_classes,
    #                         args.num_layers, args.batchnorm, args.dropout, args.activation)
    # else:
    #     raise NotImplementedError

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