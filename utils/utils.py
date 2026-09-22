from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import random
import math
from tqdm import tqdm, trange
from pybloom_live import BloomFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch2pad(batch):
    '''
    The j-th element in batch vector is i if node j is in the i-th subgraph.
    The i-th row of pad matrix contains the nodes in the i-th subgraph.
    batch [0,1,0,0,1,1,2,2]->pad [[0,2,3],[1,4,5],[6,7,-1]]
    '''
    uni, inv = batch.unique(return_inverse=True)
    idx = torch.arange(inv.shape[0], device=batch.device)
    return pad_sequence([idx[batch == i] for i in uni[uni >= 0]],
                        batch_first=True,
                        padding_value=-1).to(torch.int64)


@torch.jit.script
def pad2batch(pad):
    '''
    pad [[0,2,3],[1,4,5],[6,7,-1]]->batch [0,1,0,0,1,1,2,2]
    '''
    batch = torch.arange(pad.shape[0])
    batch = batch.reshape(-1, 1)
    batch = batch[:, torch.zeros(pad.shape[1], dtype=torch.int64)]
    batch = batch.to(pad.device).flatten()
    pos = pad.flatten()
    idx = pos >= 0
    return batch[idx], pos[idx]


@torch.jit.script
def MaxZOZ(x, pos):
    '''
    produce max-zero-one label
    x is node feature
    pos is a pad matrix like [[0,2,3],[1,4,5],[6,7,-1]], whose i-th row contains the nodes in the i-th subgraph.
    -1 is padding value.
    '''
    z = torch.zeros(x.shape[0], device=x.device, dtype=torch.int64)
    pos = pos.flatten()
    # pos[pos >= 0] removes -1 from pos
    tpos = pos[pos >= 0].to(z.device)
    z[tpos] = 1
    return z

def sync_shuffle(sample_list, max_num=-1):
    index = torch.randperm(len(sample_list[0]))
    if max_num > 0:
        index = index[:max_num]
    new_list = []
    for s in sample_list:
        new_list.append(s[index])
    return new_list

def np2tensor_hyper(vec, dtype):
    vec = np.asarray(vec)
    # vec = np.asarray([x.cpu() for x in vec])
    if len(vec.shape) == 1:
        return [torch.as_tensor(v, dtype=dtype) for v in vec]
    else:
        return torch.as_tensor(vec, dtype=dtype)

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

def neighbor_check(temp, dict):
    return tuple(temp) in dict


def sync_shuffle(sample_list, max_num=-1):
    index = torch.randperm(len(sample_list[0]))
    if max_num > 0:
        index = index[:max_num]
    new_list = []
    for s in sample_list:
        new_list.append(s[index])
    return new_list


def build_hash(data, min_size, max_size, capacity=None):
    if capacity is None:
        capacity = len(data) * 5
        capacity = int(math.ceil(capacity)) + 1000
        print("total_capacity", capacity)
    dict_list = []
    for i in range(max_size + 1):
        if i < min_size:
            dict_list.append(BloomFilter(10, 1e-3))
        else:
            dict_list.append(BloomFilter(capacity, 1e-3))

    print(len(dict_list))
    for datum in tqdm(data):
        if len(datum) > max_size:
            continue
        dict_list[len(datum)].add(tuple(datum))

    print(len(dict_list[min_size]) / dict_list[min_size].capacity)

    print(len(dict_list[-1]))
    length_list = [len(dict_list[i]) for i in range(len(dict_list))]
    print(length_list)

    return dict_list


####### negative samples generation ########
def neighbor_check(temp, data_dict):
    return tuple(temp) in data_dict

### 严格挑选负样本
def generate_strict_negative(x, dict1, weight=None, neg_num=3, min_dis=5):
    if len(weight) == 0:
        weight = torch.ones(len(x), dtype=torch.float)

    min_size, max_size = min([len(j) for j in x]), max([len(j) for j in x])
    if len(weight) == 0:
        weight = torch.ones(len(x), dtype=torch.float)

    change_num_list = [[] for i in range(max_size + 1)]
    for s in range(min_size, max_size + 1):
        change_num = np.random.binomial(
                            s, 0.5, int(len(x) * (math.ceil(neg_num) * 2)))
        change_num = change_num[change_num != 0]

        change_num_list[s] = list(change_num)

    neg_list = []
    new_x = []
    new_index = []
    neg_weight = []
    size_list = []
    size_neg_list = []
    
    for j, sample in enumerate(x):
        for i in range(int(math.ceil(neg_num))):

            decompose_sample = np.copy(sample)
            list1 = change_num_list[decompose_sample.shape[-1]]
            change_num = list1.pop()
            changes = np.random.choice(
                np.arange(decompose_sample.shape[-1]), change_num, replace=False)
            temp = np.copy(decompose_sample)
            trial = 0

            while neighbor_check(temp, dict1[(len(temp))]):
                temp = np.copy(decompose_sample)
                # trial += 1
                # if trial >= 10000:
                # 	temp = ""
                # 	break

                for change in changes:
                    if temp[change] not in node2chrom:
                        print(temp, decompose_sample)
                    chrom = node2chrom[temp[change]]
                    start, end = chrom_range[chrom]

                    temp[change] = int(
                        math.floor(
                            (end - start) * random.random())) + start

                temp = list(set(temp))

                if len(temp) < len(decompose_sample):
                    temp = np.copy(decompose_sample)
                    continue

                temp.sort()
                dis_list = []
                for k in range(len(temp) - 1):
                    dis_list.append(temp[k + 1] - temp[k])
                if np.min(dis_list) <= min_dis:
                    temp = np.copy(decompose_sample)

            if i == 0:
                size_list.append(len(decompose_sample))
            if len(temp) > 0:
                neg_list.append(temp)
                size_neg_list.append(len(temp))
                neg_weight.append(weight[j])
    
    pos_weight = weight
    pos_weight = torch.tensor(pos_weight).to(config.device)
    size_list = torch.tensor(size_list + size_neg_list)
    pos_part = utils.np2tensor_hyper(list(x), dtype=torch.long)
    neg = utils.np2tensor_hyper(neg_list, dtype=torch.long)
    if type(pos_part) == list:
        pos_part = pad_sequence(pos_part, batch_first=True, padding_value=0)
        neg = pad_sequence(neg, batch_first=True, padding_value=0)

    if len(neg) == 0:
        neg = torch.zeros(
            (1, pos_part.shape[-1]), dtype=torch.long, device=config.device)
    pos_part = pos_part.to(config.device)
    neg = neg.to(config.device)

    y = torch.cat([torch.ones((len(pos_part), 1), device=config.device),
                    torch.zeros((len(neg), 1), device=config.device)], dim=0)
    w = torch.cat([torch.ones((len(pos_part), 1), device=config.device) * pos_weight.view(-1, 1),
                    torch.ones((len(neg), 1), device=config.device)])
    x = torch.cat([pos_part, neg])

    return x, y, w, size_list


### 随机挑选负样本
def generate_random_negative(x, ei, ea, pos, z, y, weight, dict1, neg_num=3):
    ## random select negative samples.

    if len(weight) == 0:
        weight = torch.ones(len(x), dtype=torch.float)

    min_size, max_size = min([len(j) for j in pos]), max([len(j) for j in pos])
    if len(weight) == 0:
        weight = torch.ones(len(pos), dtype=torch.float)

    nodes = np.arange(node_num)

    neg_list = []
    neg_weight = []
    size_list = []
    size_neg_list = []

    for j, sample in enumerate(pos):
        # sample = np.copy(sample)
        sample = np.copy(sample.cpu().numpy())
        for i in range(int(math.ceil(neg_num))):

            for trial in range(500):
                neg = sorted(np.random.choice(nodes, sample.shape[-1]))
                if not neighbor_check(neg, dict1[len(sample)]):
                    neg_list.append(neg)
                    size_neg_list.append(len(neg))
                    neg_weight.append(weight[j])

                    if i == 0:
                        size_list.append(len(sample))

                    break
    
    pos_weight = weight
    # pos_weight = torch.tensor(pos_weight).to(config.device)
    size_list = torch.tensor(size_list + size_neg_list)
    pos_part = utils.np2tensor_hyper(list(pos.cpu()), dtype=torch.long)
    neg = utils.np2tensor_hyper(neg_list, dtype=torch.long)

    if type(pos_part) == list:
        pos_part = pad_sequence(pos_part, batch_first=True, padding_value=0)
        neg = pad_sequence(neg, batch_first=True, padding_value=0)

    if len(neg) == 0:
        neg = torch.zeros(
            (1, pos_part.shape[-1]), dtype=torch.long, device=config.device)
    pos_part = pos_part.to(config.device)
    neg = neg.to(config.device)

    y = torch.cat([torch.ones((len(pos_part), 1), device=config.device),
                    torch.zeros((len(neg), 1), device=config.device)], dim=0)
    w = torch.cat([torch.ones((len(pos_part), 1), device=config.device) * pos_weight.view(-1, 1),
                    torch.ones((len(neg), 1), device=config.device)])
    pos = torch.cat([pos_part, neg])

    z = utils.MaxZOZ(x, pos)

    return x, ei, ea, pos, z, y, w
