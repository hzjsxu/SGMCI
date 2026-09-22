import torch
from tqdm import tqdm, trange
import numpy as np


# def train(optimizer, model, dataloader, loss_fn):
#     '''
#     Train models in an epoch.
#     '''
#     model.train()
#     total_loss = []
#     '''
#     batch[0]: x
#     batch[1]: edge_index
#     batch[2]: edge_attr / edge_weight
#     batch[3]: pos
#     batch[4]: z / node label (labeling trick) ***
#     batch[5]: y (label)
#     batch[6]: subgraph weight
#     '''
#     alpha = 0.5
#     beta = 0.1
#     for batch in dataloader:
#         optimizer.zero_grad()
#         sub_G_weight = batch[-1]
#         pred, recon_loss = model(*batch[:-2], id=0)
#         bce_loss = loss_fn(pred, batch[-2], weight=sub_G_weight)  ## loss_fn 可以添加weight
#         loss = bce_loss * alpha + recon_loss * beta
#         loss.backward()
#         total_loss.append(loss.detach().item())
#         optimizer.step()
#     return sum(total_loss) / len(total_loss)


def train(optimizer, model, dataloader, loss_fn, metrics):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    preds = []
    ys = []
    size_list = []
    '''
    batch[0]: x
    batch[1]: edge_index
    batch[2]: edge_attr / edge_weight
    batch[3]: pos
    batch[4]: z / node label (labeling trick) ***
    batch[5]: y (label)
    batch[6]: subgraph weight
    '''
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        sub_G_weight = batch[-1]
        pred, node_emb = model(*batch[:-2], id=0)
        batch_s = [x[x>-1].shape[0] for x in batch[3]]

        preds.append(pred)
        ys.append(batch[-2])
        size_list.append(batch_s)
        # loss = loss_fn(pred, batch[-2], weight=sub_G_weight)  ## loss_fn 可以添加weight
        loss = loss_fn(pred, batch[-2])
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()

    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    size_list = np.concatenate(size_list)

    return metrics(pred.cpu().detach().numpy(), y.cpu().detach().numpy(), size_list), sum(total_loss) / len(total_loss), node_emb.detach().cpu().numpy()


@torch.no_grad()
def test(model, dataloader, metrics, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    total_loss = []
    preds = []
    ys = []
    sub_G_weights = []
    size_list = []

    for batch in dataloader:
        pred, node_emb = model(*batch[:-2])
        preds.append(pred)
        ys.append(batch[-2])
        sub_G_weights.append(batch[-1])
        batch_s = [x[x>-1].shape[0] for x in batch[3]]
        size_list.append(batch_s)
        # loss = loss_fn(pred, batch[-2], weight=sub_G_weight)     ## loss_fn 可以添加weight
        loss = loss_fn(pred, batch[-2])
        total_loss.append(loss.detach().item())

    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    sub_G_weight = torch.cat(sub_G_weights, dim=0)
    size_list = np.concatenate(size_list)

    return metrics(pred.cpu().numpy(), y.cpu().numpy(), size_list), sum(total_loss) / len(total_loss), y.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy(), size_list, sub_G_weight.squeeze().cpu().numpy(), node_emb.detach().cpu().numpy()


@torch.no_grad()
def test_emb(model, dataloader, metrics, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    sub_G_weights = []
    for batch in dataloader:
        pred = model(*batch[:-1])
        preds.append(pred)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)
