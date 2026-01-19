from utils import models, SubGDataset, train, metrics, utils, config
import datasets
import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import argparse
import torch.nn as nn
import functools
import numpy as np
import pandas as pd
import time
import random
import yaml
import copy

from sklearn.decomposition import PCA

import networkx as nx
import os

import warnings
warnings.filterwarnings("error")

from tqdm import tqdm, trange
import math

parser = argparse.ArgumentParser(description='')
# Dataset settings
parser.add_argument('--dataset', type=str, default='HiPore-C_K562')
# Node feature settings.
# deg means use node degree. one means use homogeneous embeddings.
# nodeid means use pretrained node embeddings in ./Emb
# seq means use precomputed kmer features in ./Emb
# epi means use precomputed epi features in ./Emb
parser.add_argument('--use_deg', action='store_true')
parser.add_argument('--use_one', action='store_true')

parser.add_argument('--use_nodeid', action='store_true')
parser.add_argument('--use_seq', action='store_true')
parser.add_argument('--use_epi', action='store_true')

parser.add_argument('--ns_mode', type=str, default='SNS')  ## --negative_sampling mode: RNS random negative sampling; BNS binomial negative sampling; SNS strict negative sampling.
parser.add_argument('--decompose', type=str, default='D')  ## --decompose data: D decompose; ND not decompose.

parser.add_argument('--genome', type=str, default='hg38')  ## --genome: hg38 mm10
parser.add_argument('--binsize', type=str, default='1Mb')  ## --binsize: 1Mb 500kb 100kb

parser.add_argument('--test_chr', type=str, default='chr1')  ## --test_chr: chr1
# parser.add_argument('--mergePool', type=str, action='store_true') ## --mergePool: pooling node embedings into subgraph embeddings using Mean, Max, Size and node distance (or node number).

parser.add_argument('--epochs', type=int, default=10, help='training epoch numbers. Default: 10')

# node label settings
parser.add_argument('--use_maxzeroone', action='store_true')

parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')

# args = parser.parse_args(['--use_seed', '--use_nodeid', '--use_maxzeroone', '--repeat', '1', '--device', '1', '--dataset', 'HiPore-C_GM12878_1Mb_O5'])
# args = parser.parse_args(['--use_seed', '--use_nodeid', '--use_maxzeroone', '--repeat', '1', '--device', '0', '--dataset', 'test_hic'])
# args = parser.parse_args(['--use_seed', '--use_nodeid', '--use_maxzeroone', '--repeat', '1', '--device', '1', '--dataset', 'test_GM12878_1Mb', '--ns_mode', 'BNS', '--decompose', 'ND', '--genome', 'hg38', '--binsize', '1Mb'])
args = parser.parse_args()
config.set_device(args.device)

def set_seed(seed: int):
    print("seed ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu

if args.use_seed:
    set_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

baseG = datasets.load_dataset(args.dataset, args.ns_mode, args.test_chr, args.decompose, args.genome, args.binsize)

trn_dataset, val_dataset, tst_dataset = None, None, None
max_deg, output_channels = 0, 1
score_fn = metrics.auroc

if baseG.y.unique().shape[0] == 2:
    # binary classification task
    def loss_fn(x, y, weight=None):
        # return BCEWithLogitsLoss()(x.flatten(), y.flatten())
        return F.binary_cross_entropy_with_logits(x.flatten(), y.flatten(), weight=weight)

    baseG.y = baseG.y.to(torch.float)
    if baseG.y.ndim > 1:
        output_channels = baseG.y.shape[1]
    else:
        output_channels = 1
    # score_fn = metrics.binaryf1
    score_fn = metrics.auroc
    # score_fn = metrics.aupr

else:
    # multi-class classification task
    baseG.y = baseG.y.to(torch.int64)
    loss_fn = CrossEntropyLoss()
    output_channels = baseG.y.unique().shape[0]
    score_fn = metrics.microf1

loader_fn = SubGDataset.GDataloader
tloader_fn = SubGDataset.GDataloader

def split():
    '''
    load and split dataset.
    '''
    # initialize and split dataset
    global trn_dataset, val_dataset, tst_dataset, baseG
    global max_deg, output_channels, loader_fn, tloader_fn

    baseG = datasets.load_dataset(args.dataset, args.ns_mode, args.test_chr, args.decompose, args.genome, args.binsize)
    if baseG.y.unique().shape[0] == 2:
        baseG.y = baseG.y.to(torch.float)
    else:
        baseG.y = baseG.y.to(torch.int64)
    # initialize node features
    if args.use_deg:
        baseG.setDegreeFeature()
    elif args.use_one:
        baseG.setOneFeature()
    elif args.use_nodeid:
        baseG.setNodeIdFeature()
    else:
        raise NotImplementedError

    max_deg = torch.max(baseG.x)
    baseG.to(config.device)

    # split data
    trn_dataset = SubGDataset.GDataset(*baseG.get_split("train"))
    val_dataset = SubGDataset.GDataset(*baseG.get_split("valid"))
    tst_dataset = SubGDataset.GDataset(*baseG.get_split("test"))

    # choice of dataloader
    if args.use_maxzeroone:
        ## i.e., means GLASS with labeling trick
        def tfunc(ds, bs, shuffle=True, drop_last=True):
            return SubGDataset.ZGDataloader(ds,
                                            bs,
                                            z_fn=utils.MaxZOZ,
                                            shuffle=shuffle,
                                            drop_last=drop_last)

        def loader_fn(ds, bs):
            return tfunc(ds, bs)

        def tloader_fn(ds, bs):
            return tfunc(ds, bs, True, False)
    else:
        ## i.e., GNN-plain: means GLASS without labeling trick
        def loader_fn(ds, bs):
            return SubGDataset.GDataloader(ds, bs)

        def tloader_fn(ds, bs):
            return SubGDataset.GDataloader(ds, bs, shuffle=True)


def buildModel(hidden_dim, conv_layer, dropout, jk, pool, z_ratio, aggr):
    '''
    Build a GLASS model.
    Args:
        jk: whether to use Jumping Knowledge Network.
        conv_layer: number of GLASSConv.
        pool: pooling function transfer node embeddings to subgraph embeddings.
        z_ratio: see GLASSConv in impl/model.py. Z_ratio in [0.5, 1].
        aggr: aggregation method. mean, sum, or gcn. 
    '''

    # use pretrained node embeddings.
    # if args.use_nodeid:
    print("load ", f"./Emb/{args.dataset}_{hidden_dim}.pt")
    emb = torch.load(f"./Emb/{args.dataset}_{hidden_dim}.pt", map_location=torch.device('cpu')).detach()
        
    if args.use_epi:
        name = "_".join(args.dataset.split('_')[:-2])
        chr_name = "_".join(args.dataset.split('_')[:-1])
        print("load ", f"./Emb/{name}_feature/{chr_name}_epi8.pt")

        emb_epi = torch.load(f"./Emb/{name}_feature/{chr_name}_epi8.pt", map_location=torch.device('cpu')).detach()
        emb = torch.cat([emb, emb_epi], dim=1).to(torch.float32)
    
    if args.use_seq:
        name = "_".join(args.dataset.split('_')[:-2])
        chr_name = "_".join(args.dataset.split('_')[:-1])
        print("load ", f"./Emb/{name}_feature/{chr_name}_seq1344.pt")

        emb_seq = torch.load(f"./Emb/{name}_feature/{chr_name}_seq1344.pt", map_location=torch.device('cpu')).detach()
        emb = torch.cat([emb, emb_seq], dim=1).to(torch.float32)

        # pca=PCA(n_components=64)
        # emb = pca.fit_transform(emb)
        # emb = torch.from_numpy(emb).to(torch.float32)

    conv = models.EmbZGConv(emb.shape[1],
                            hidden_dim,
                            conv_layer,
                            max_deg=max_deg,
                            activation=nn.ELU(inplace=True),
                            jk=jk,
                            dropout=dropout,
                            conv=functools.partial(models.GLASSConv,
                                                    aggr=aggr,
                                                    z_ratio=z_ratio,
                                                    dropout=dropout),
                            gn=True)  ##

    conv.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)

    # mlp = nn.Linear(hidden_dim * (conv_layer) if jk else hidden_dim, output_channels)

    ## 改为加2层MLP
    mlp1 = nn.Linear(hidden_dim * (conv_layer) * 3 + 2 if jk else hidden_dim * 3 + 2, hidden_dim * (conv_layer) if jk else hidden_dim)
    # mlp1 = nn.Linear(hidden_dim * (conv_layer) * 3  if jk else hidden_dim * 3 , hidden_dim * (conv_layer) if jk else hidden_dim)
    mlp2 = nn.Linear(hidden_dim * (conv_layer) if jk else hidden_dim, output_channels)

    pool_fn_fn = {
        "mean": models.MeanPool,
        "max": models.MaxPool,
        "sum": models.AddPool,
        "size": models.SizePool
    }
    if pool in pool_fn_fn:
        pool_fn1 = pool_fn_fn[pool]()
    else:
        raise NotImplementedError

    # gnn = models.GLASS(conv, 
    #                     torch.nn.ModuleList([mlp]),
    #                     torch.nn.ModuleList([pool_fn1])
    #                     ).to(config.device)

    gnn = models.GLASS(conv, 
                    torch.nn.ModuleList([mlp1, mlp2]),
                    torch.nn.ModuleList([pool_fn1])
                    ).to(config.device)

    return gnn


#####################################################################
# @torch.no_grad()
# def save_embeddings(model, trn_dataset):
#     ## save node embeddings.
#     model.eval()
#     emb = model.NodeEmb(trn_dataset.x, trn_dataset.edge_index, trn_dataset.edge_weight)
#     return emb

#####################################################################
def test(pool="size",
        aggr="mean",
        hidden_dim=64,
        conv_layer=8,
        dropout=0.3,
        jk=1,
        lr=1e-3,
        z_ratio=0.8,
        batch_size=None,
        resi=0.7,
        size_list=None):
    
    # tst_baseG = datasets.load_dataset_for_test(args.dataset, chrom=args.test_chr)
    # # we set batch_size = tst_dataset.y.shape[0] // num_div.
    # num_div = tst_baseG.y.shape[0] / batch_size

    outs = []
    t1 = time.time()
    # we set batch_size = tst_dataset.y.shape[0] // num_div.
    num_div = tst_dataset.y.shape[0] / batch_size

    for repeat in range(args.repeat):
        set_seed((1 << repeat) - 1)
        print(f'repeat {repeat}')

        model = buildModel(hidden_dim, conv_layer, dropout, jk, pool, z_ratio, aggr)  ### build model

        # trn_loader = loader_fn(trn_dataset, batch_size)
        # val_loader = tloader_fn(val_dataset, batch_size)
        # tst_loader = tloader_fn(tst_dataset, batch_size)

        optimizer = Adam(model.parameters(), lr=lr)
        scd = lr_scheduler.ReduceLROnPlateau(optimizer, factor=resi, min_lr=5e-5)

        val_score = 0
        tst_score = 0
        early_stop = 0
        trn_time = []

        epoch_trn_loss = []
        epoch_tst_loss = []

        for i in range(args.epochs):
            trn_loader = loader_fn(trn_dataset, batch_size)
            val_loader = tloader_fn(val_dataset, batch_size)
            tst_loader = tloader_fn(tst_dataset, batch_size)
            t1 = time.time()
            (trn_score_str, trn_score), trn_loss, trn_emb = train.train(optimizer, model, trn_loader, loss_fn, metrics=score_fn)  ### Train
            trn_time.append(time.time() - t1)
            scd.step(trn_loss)

            # if i >= 100 / num_div:
            (score_str, score), tst_loss, val_y_label, val_y_pred, size_list, weight_list, val_emb = train.test(model, val_loader, score_fn, loss_fn=loss_fn)

            if score > val_score:
                early_stop = 0
                val_score = score
                val_score_str = score_str
                val_size_list = size_list
                val_weight_list = weight_list

                (score_str, score), tst_loss, tst_y_label, tst_y_pred, size_list, weight_list, tst_emb = train.test(model, tst_loader, score_fn, loss_fn=loss_fn)
                tst_score = score
                tst_score_str = score_str
                tst_size_list = size_list
                tst_weight_list = weight_list

                # print(f"iter {i} trn_loss {trn_loss:.4f} tst_loss {tst_loss:.4f} trn {trn_score:.4f} val {val_score:.4f} tst {tst_score:.4f}", flush=True)
                print(f"iter {i} trn_loss {trn_loss:.4f} tst_loss {tst_loss:.4f} trn {trn_score:.4f} val {val_score:.4f} tst {tst_score:.4f} trn_size_score {trn_score_str} val_size_score {val_score_str} tst_size_score {tst_score_str}", flush=True)
            # elif score >= val_score - 1e-5:
            #     score, _ = train.test(model, tst_loader, score_fn, loss_fn=loss_fn)
            #     tst_score = max(score, tst_score)
            #     print(
            #         f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {score:.4f}",
            #         flush=True)
            else:
                early_stop += 1
                print(f"===> iter {i} trn_loss {trn_loss:.4f} tst_loss {tst_loss:.4f} trn {trn_score} val {score} tst {train.test(model, tst_loader, score_fn, loss_fn=loss_fn)[0][1]}", flush=True)
                if i % 10 == 0:
                    print(f"===> iter {i} trn_loss {trn_loss:.4f} tst_loss {tst_loss:.4f} trn {trn_score} val {score} tst {train.test(model, tst_loader, score_fn, loss_fn=loss_fn)[0][1]}", flush=True)

            if val_score >= 1 - 1e-5:
                early_stop += 1
            # if early_stop > 100 / num_div:
            if early_stop > 3:
                break

            epoch_trn_loss.append(trn_loss)
            epoch_tst_loss.append(tst_loss)

        print(
            f"end: epoch {i+1}, train time {sum(trn_time):.2f} s, val {val_score}, tst {tst_score}", flush=True)
        ## save results.
        checkpoint = copy.deepcopy(model)
        # torch.save(checkpoint, f'./results/{args.dataset}/{args.dataset}_r{repeat}.pt')

        if args.use_epi & args.use_seq:
            pd.DataFrame({'Size': tst_size_list, 'Weight': tst_weight_list, 'Pred': tst_y_pred, 'Label': tst_y_label}).to_csv(f'{out_dir}/{prefix}_StrucEpiSeq_r{repeat}.tsv', sep='\t', header=True, index=False)
            pd.DataFrame({'trn_loss': epoch_trn_loss, 'tst_loss': epoch_tst_loss}).to_csv( f'{out_dir}/{prefix}_StrucEpiSeq_r{repeat}.epoch_loss.tsv', sep='\t', index=False, header=False)
            np.save(f'{out_dir}/{prefix}_StrucEpiSeq_r{repeat}.trn_embeddings.npy', trn_emb)
            np.save(f'{out_dir}/{prefix}_StrucEpiSeq_r{repeat}.val_embeddings.npy', val_emb)
            np.save(f'{out_dir}/{prefix}_StrucEpiSeq_r{repeat}.tst_embeddings.npy', tst_emb)
            torch.save(checkpoint, f'{out_dir}/{prefix}_StrucEpiSeq_r{repeat}.pt')
        elif args.use_epi:
            pd.DataFrame({'Size': tst_size_list, 'Weight': tst_weight_list, 'Pred': tst_y_pred, 'Label': tst_y_label}).to_csv(f'{out_dir}/{prefix}_StrucEpi_r{repeat}.tsv', sep='\t', header=True, index=False)
            pd.DataFrame({'trn_loss': epoch_trn_loss, 'tst_loss': epoch_tst_loss}).to_csv( f'{out_dir}/{prefix}_StrucEpi_r{repeat}.epoch_loss.tsv', sep='\t', index=False, header=False)
            np.save(f'{out_dir}/{prefix}_StrucEpi_r{repeat}.trn_embeddings.npy', trn_emb)
            np.save(f'{out_dir}/{prefix}_StrucEpi_r{repeat}.val_embeddings.npy', val_emb)
            np.save(f'{out_dir}/{prefix}_StrucEpi_r{repeat}.tst_embeddings.npy', tst_emb)
            torch.save(checkpoint, f'{out_dir}/{prefix}_StrucEpi_r{repeat}.pt')
        elif args.use_seq:
            pd.DataFrame({'Size': tst_size_list, 'Weight': tst_weight_list, 'Pred': tst_y_pred, 'Label': tst_y_label}).to_csv(f'{out_dir}/{prefix}_StrucSeq_r{repeat}.tsv', sep='\t', header=True, index=False)
            pd.DataFrame({'trn_loss': epoch_trn_loss, 'tst_loss': epoch_tst_loss}).to_csv( f'{out_dir}/{prefix}_StrucSeq_r{repeat}.epoch_loss.tsv', sep='\t', index=False, header=False)
            np.save(f'{out_dir}/{prefix}_StrucSeq_r{repeat}.trn_embeddings.npy', trn_emb)
            np.save(f'{out_dir}/{prefix}_StrucSeq_r{repeat}.val_embeddings.npy', val_emb)
            np.save(f'{out_dir}/{prefix}_StrucSeq_r{repeat}.tst_embeddings.npy', tst_emb)
            torch.save(checkpoint, f'{out_dir}/{prefix}_StrucSeq_r{repeat}.pt')
        else:
            pd.DataFrame({'Size': tst_size_list, 'Weight': tst_weight_list, 'Pred': tst_y_pred, 'Label': tst_y_label}).to_csv(f'{out_dir}/{prefix}_Struc_r{repeat}.tsv', sep='\t', header=True, index=False)
            pd.DataFrame({'trn_loss': epoch_trn_loss, 'tst_loss': epoch_tst_loss}).to_csv( f'{out_dir}/{prefix}_Struc_r{repeat}.epoch_loss.tsv', sep='\t', index=False, header=True)
            np.save(f'{out_dir}/{prefix}_Struc_r{repeat}.trn_embeddings.npy', trn_emb)
            np.save(f'{out_dir}/{prefix}_Struc_r{repeat}.val_embeddings.npy', val_emb)
            np.save(f'{out_dir}/{prefix}_Struc_r{repeat}.tst_embeddings.npy', tst_emb)
            torch.save(checkpoint, f'{out_dir}/{prefix}_Struc_r{repeat}.pt')
        outs.append(tst_score)

    print(
        f"average {np.average(outs):.4f} error {np.std(outs) / np.sqrt(len(outs)):.4f}"
    )

print(args)
###################  read configuration  #################
with open(f"config/{args.dataset}.yml") as f:
    params = yaml.safe_load(f)

print("params", params, flush=True)
split()

out_dir = f'./results/{args.dataset}/{args.dataset}_test_{args.test_chr}'
prefix = f'{args.dataset}_test_{args.test_chr}_{args.decompose}_{args.ns_mode}'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

test(**(params))