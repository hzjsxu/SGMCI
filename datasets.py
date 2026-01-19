import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot
import torch
from torch_geometric.utils import is_undirected, to_undirected, negative_sampling, to_networkx
from torch_geometric.data import Data
import networkx as nx
import os
from utils.utils import build_hash

class BaseGraph(Data):
    def __init__(self, x, edge_index, edge_weight, subG_node, subG_label, subG_weight, mask):
        '''
        A general format for datasets.
        Args:
            x: node feature. For our used datasets, x is empty vector.
            subG_node: a matrix like [[0,2,3],[1,4,5],[6,7,-1]], whose i-th row contains the nodes in the i-th subgraph. -1 is for padding.
            subG_label: the target of subgraphs.
            mask: of shape (number of subgraphs), type torch.long. mask[i]=0,1,2 if i-th subgraph is in the training set, validation set and test set respectively. 
        '''
        super(BaseGraph, self).__init__(x=x,
                                        edge_index=edge_index,
                                        edge_attr=edge_weight,
                                        pos=subG_node,
                                        y=subG_label)
        self.subG_weight = subG_weight
        self.mask = mask
        self.to_undirected()

    def setDegreeFeature(self, mod=1):
        # use node degree as node features.
        adj = torch.sparse_coo_tensor(self.edge_index, self.edge_attr,
                                        (self.x.shape[0], self.x.shape[0]))
        degree = torch.sparse.sum(adj, dim=1).to_dense().to(torch.int64)
        degree = torch.div(degree, mod, rounding_mode='floor')
        degree = torch.unique(degree, return_inverse=True)[1]
        self.x = degree.reshape(self.x.shape[0], 1, -1)

    def setOneFeature(self):
        # use homogeneous node features.
        self.x = torch.ones((self.x.shape[0], 1, 1), dtype=torch.int64)

    def setNodeIdFeature(self):
        # use nodeid as node features.
        self.x = torch.arange(self.x.shape[0], dtype=torch.int64).reshape(
            self.x.shape[0], 1, -1)

    def get_split(self, split: str):
        tar_mask = {"train": 0, "valid": 1, "test": 2}[split]
        return self.x, self.edge_index, self.edge_attr, self.pos[
            self.mask == tar_mask], self.y[self.mask == tar_mask], self.subG_weight[self.mask == tar_mask]

    def to_undirected(self):
        if not is_undirected(self.edge_index):
            self.edge_index, self.edge_attr = to_undirected(
                self.edge_index, self.edge_attr)

    def get_LPdataset(self, use_loop=False):
        # generate link prediction dataset for pretraining GNNs
        neg_edge = negative_sampling(self.edge_index)
        x = self.x
        ei = self.edge_index
        ea = self.edge_attr
        pos = torch.cat((self.edge_index, neg_edge), dim=1).t()
        y = torch.cat((torch.ones(ei.shape[1]),
                        torch.zeros(neg_edge.shape[1]))).to(ei.device)

        return x, ei, ea, pos, y

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.pos = self.pos.to(device)
        self.y = self.y.to(device)
        self.mask = self.mask.to(device)
        self.subG_weight = self.subG_weight.to(device)
        return self

# ### load dataset from raw data.
# def load_dataset(name: str, min_freq=11, size_list=[3], use_preEmb=False, hidden_dim=64):
#     if name in ["hic", "K562_HiPoreC_ind", "HiPore-C_GM12878"]:
#         # copied from https://github.com/mims-harvard/SubGNN/blob/main/SubGNN/subgraph_utils.py
#         def read_subgraphs(sub_f, sub_w_f, min_freq):

#             data = np.load(sub_f, allow_pickle=True)
#             data = data - 1   ## minus - 1 because node starts with 1
#             data_weight = np.load(sub_w_f, allow_pickle=True)

#             ## filter by min_freq
#             data = data[data_weight >= min_freq]
#             data_weight = data_weight[data_weight >= min_freq]

#             ## build hash dict.
#             min_size, max_size = int(np.min(size_list)), int(np.max(size_list))
#             data_dict = build_hash(data, max_size=max_size, min_size=min_size)

#             ## split data into train/validation/test = 8/1/1
#             index = np.arange(len(data))
#             np.random.shuffle(index)
#             train_split = int(0.8 * len(index))
#             val_split = int(0.1 * len(index))

#             trn_data = data[index[:train_split]].tolist()
#             val_data = data[index[train_split:(train_split + val_split)]].tolist()
#             tst_data = data[index[(train_split + val_split): ]].tolist()

#             trn_data_weight = data_weight[index[:train_split]]
#             val_data_weight = data_weight[index[train_split:(train_split + val_split)]]
#             tst_data_weight = data_weight[index[(train_split + val_split): ]]

#             trn_data_label = np.ones(len(trn_data_weight))
#             val_data_label = np.ones(len(val_data_weight))
#             tst_data_label = np.ones(len(tst_data_weight))

#             trn_data_label = torch.tensor(trn_data_label).squeeze()
#             val_data_label = torch.tensor(val_data_label).squeeze()
#             tst_data_label = torch.tensor(tst_data_label).squeeze()

#             trn_data_weight = torch.tensor(trn_data_weight).squeeze()
#             val_data_weight = torch.tensor(val_data_weight).squeeze()
#             tst_data_weight = torch.tensor(tst_data_weight).squeeze()

#             return trn_data, trn_data_weight, trn_data_label, val_data, val_data_weight, val_data_label, tst_data, tst_data_weight, tst_data_label, data_dict


#         trn_sub_G, trn_sub_G_weight, trn_sub_G_label, val_sub_G, val_sub_G_weight, val_sub_G_label, tst_sub_G, tst_sub_G_weight, tst_sub_G_label, data_dict \
#             = read_subgraphs(sub_f=f'/data/xujs/Project/DeepLearning/MCIP/Results/{name}/1000kb/all_3_subgraph.npy',
#                             sub_w_f=f'/data/xujs/Project/DeepLearning/MCIP/Results/{name}/1000kb/all_3_subgraph_freq.npy',
#                             min_freq=min_freq)

#         mask = torch.cat(
#             (torch.zeros(len(trn_sub_G_weight), dtype=torch.int64),
#             torch.ones(len(val_sub_G_weight), dtype=torch.int64),
#              2 * torch.ones(len(tst_sub_G_weight))),
#             dim=0)
        
#         sub_G_label = torch.cat((trn_sub_G_label, val_sub_G_label, tst_sub_G_label))
        
#         sub_G_weight = torch.cat((trn_sub_G_weight, val_sub_G_weight, tst_sub_G_weight))

#         pos = pad_sequence(
#                 [torch.tensor(i) for i in trn_sub_G + val_sub_G + tst_sub_G],
#                 batch_first=True,
#                 padding_value=-1)
        
#         # Edge list.
#         rawedge = nx.read_edgelist(f"/data/xujs/Project/DeepLearning/MCIP/Results/{name}/1000kb/edge_list.txt", nodetype=int, data=(('weight',float),)).edges(data=True)
#         edge_index = torch.tensor([[int(i[0]), int(i[1])]
#                                     for i in rawedge]).t()
#         edge_weight = torch.tensor([i[2]['weight'] 
#                                     for i in rawedge])
#         num_node = max([torch.max(pos), torch.max(edge_index)]) + 1

#         if use_preEmb:
#             print("load pretrained embedding:", f"/data/xujs/Project/HiC2PoreC/code/SGMCI/Emb/{name}/1000kb_{hidden_dim}.pt")
#             x = torch.load(f"/data/xujs/Project/HiC2PoreC/code/GLASS_ind/Emb/{name}/1000kb_{hidden_dim}.pt", map_location=torch.device('cpu')).detach()
#         else:
#             # x = torch.empty((num_node, 1, 0))
#             x = torch.arange(num_node, dtype=torch.int64).reshape(num_node, 1, -1)

#         # return x, edge_index, edge_weight, pos, sub_G_label, mask, sub_G_weight
#         return BaseGraph(x, edge_index, edge_weight, pos, sub_G_label, sub_G_weight, mask), data_dict
#         # return BaseGraph(x, edge_index, torch.ones(edge_index.shape[1]), pos, label.to(torch.float), mask)
#         # return BaseGraph(x, edge_index, edge_weight, pos, label.to(torch.float), mask, sub_G_weight)
    
#     else:
#         raise NotImplementedError()


### load dataset from precossed subgraph data.
## original data loading style.
def load_dataset(name: str, ns_mode: str, test_chr: str, decompose: str, genome: str, binsize: str):

    # copied from https://github.com/mims-harvard/SubGNN/blob/main/SubGNN/subgraph_utils.py
    def read_subgraphs(sub_f, split=True):
        label_idx = 0
        labels = {}
        train_sub_G, val_sub_G, test_sub_G = [], [], []
        train_sub_G_label, val_sub_G_label, test_sub_G_label = [], [], []
        train_sub_G_weight, val_sub_G_weight, test_sub_G_weight = [], [], []
        train_mask, val_mask, test_mask = [], [], []

        # Parse data
        with open(sub_f) as fin:
            subgraph_idx = 0
            for line in fin:
                nodes = [
                    int(n) for n in line.split("\t")[0].split("-")
                    if n != ""
                ]
                if len(nodes) != 0:
                    if len(nodes) == 1:
                        print(nodes)
                    l = line.split("\t")[1].split("-")
                    if len(l) > 1:
                        multilabel = True
                    for lab in l:
                        if lab not in labels.keys():
                            labels[lab] = label_idx
                            label_idx += 1

                    w = [int(float(line.split("\t")[-1].strip()))]

                    if line.split("\t")[2].strip() == "train":
                        train_sub_G.append(nodes)
                        train_sub_G_label.append(
                            [labels[lab] for lab in l])
                        train_sub_G_weight.append(w)
                        train_mask.append(subgraph_idx)
                    elif line.split("\t")[2].strip() == "val":
                        val_sub_G.append(nodes)
                        val_sub_G_label.append([labels[lab] for lab in l])
                        val_sub_G_weight.append(w)
                        val_mask.append(subgraph_idx)
                    elif line.split("\t")[2].strip() == "test":
                        test_sub_G.append(nodes)
                        test_sub_G_label.append([labels[lab] for lab in l])
                        test_sub_G_weight.append(w)
                        test_mask.append(subgraph_idx)

                    subgraph_idx += 1

        train_sub_G_label = torch.tensor(train_sub_G_label).squeeze()
        val_sub_G_label = torch.tensor(val_sub_G_label).squeeze()
        test_sub_G_label = torch.tensor(test_sub_G_label).squeeze()

        train_sub_G_weight = torch.tensor(train_sub_G_weight).squeeze()
        val_sub_G_weight = torch.tensor(val_sub_G_weight).squeeze()
        test_sub_G_weight = torch.tensor(test_sub_G_weight).squeeze()

        if len(val_mask) < len(test_mask):
            return train_sub_G, train_sub_G_label, train_sub_G_weight, test_sub_G, test_sub_G_label, test_sub_G_weight, val_sub_G, val_sub_G_label, val_sub_G_weight

        return train_sub_G, train_sub_G_label, train_sub_G_weight, val_sub_G, val_sub_G_label, val_sub_G_weight, test_sub_G, test_sub_G_label, test_sub_G_weight


    train_sub_G, train_sub_G_label, train_sub_G_weight, val_sub_G, val_sub_G_label, val_sub_G_weight, test_sub_G, test_sub_G_label, test_sub_G_weight = read_subgraphs(
            f"./dataset/{name}/test_{test_chr}_subgraphs/{decompose}_{ns_mode}_subgraphs.pth")

    mask = torch.cat(
        (torch.zeros(len(train_sub_G_label), dtype=torch.int64),
        torch.ones(len(val_sub_G_label), dtype=torch.int64),
            2 * torch.ones(len(test_sub_G_label))),
        dim=0)

    label = torch.cat((train_sub_G_label, val_sub_G_label, test_sub_G_label))

    sub_G_weight = torch.cat((train_sub_G_weight, val_sub_G_weight, test_sub_G_weight))

    pos = pad_sequence(
        [torch.tensor(i) for i in train_sub_G + val_sub_G + test_sub_G],
        batch_first=True,
        padding_value=-1)

    rawedge = nx.read_edgelist(f"./dataset/{name}/edge_list.txt", nodetype=int, data=(('weight',float),)).edges(data=True)
    edge_index = torch.tensor([[int(i[0]), int(i[1])]
                                for i in rawedge]).t()
    edge_weight = torch.tensor([i[2]['weight'] 
                                for i in rawedge])

    ## 需要指定节点数量
    # df_node_num = pd.read_table('/data/xujs/Project/HiC2PoreC/code/SGMCI/dataset/hg38.1Mb.node_num.txt')
    df_node_num = pd.read_table(f'./dataset/{genome}.{binsize}.node_num.txt')
    node_num_dict = dict(zip(df_node_num['chrom'], df_node_num['chrom_node_num']))
    chrom_symbol = name.split('_')[-2]
    chrom_node_num = node_num_dict[chrom_symbol] if chrom_symbol in node_num_dict else sum(list(node_num_dict.values())) ##低分辨率下不分染色体，而是全基因组

    num_node = max([torch.max(pos), torch.max(edge_index)]) + 1
    num_node = max([num_node, chrom_node_num])
    x = torch.empty((num_node, 1, 0))

    return BaseGraph(x, edge_index, edge_weight, pos, label.to(torch.float), sub_G_weight, mask)
