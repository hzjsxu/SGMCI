import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
from torch_geometric.nn.glob.glob import global_mean_pool, global_add_pool, global_max_pool
from .utils import pad2batch

from torch_geometric.nn.pool import SAGPooling

class Seq(nn.Module):
    ''' 
    An extension of nn.Sequential. 
    Args: 
        modlist an iterable of modules to add.
    '''
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


class MLP(nn.Module):
    '''
    Multi-Layer Perception.
    Args:
        tail_activation: whether to use activation function at the last layer.
        activation: activation function.
        gn: whether to use GraphNorm layer.
    '''
    def __init__(self,
                input_channels: int,
                hidden_channels: int,
                output_channels: int,
                num_layers: int,
                dropout=0,
                tail_activation=False,
                activation=nn.ReLU(inplace=True),
                gn=False):
        super().__init__()
        modlist = []
        self.seq = None
        if num_layers == 1:
            modlist.append(nn.Linear(input_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)
        else:
            modlist.append(nn.Linear(input_channels, hidden_channels))
            for _ in range(num_layers - 2):
                if gn:
                    modlist.append(GraphNorm(hidden_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
                modlist.append(nn.Linear(hidden_channels, hidden_channels))
            if gn:
                modlist.append(GraphNorm(hidden_channels))
            if dropout > 0:
                modlist.append(nn.Dropout(p=dropout, inplace=True))
            modlist.append(activation)
            modlist.append(nn.Linear(hidden_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)

    def forward(self, x):
        return self.seq(x)


def buildAdj(edge_index, edge_weight, n_node: int, aggr: str):
    '''
        Calculating the normalized adjacency matrix.
        Args:
            n_node: number of nodes in graph.
            aggr: the aggregation method, can be "mean", "sum" or "gcn".
        '''
    adj = torch.sparse_coo_tensor(edge_index,
                                    edge_weight,
                                    size=(n_node, n_node))
    deg = torch.sparse.sum(adj, dim=(1, )).to_dense().flatten()
    deg[deg < 0.5] += 1.0
    if aggr == "mean":
        deg = 1.0 / deg
        return torch.sparse_coo_tensor(edge_index,
                                        deg[edge_index[0]] * edge_weight,
                                        size=(n_node, n_node))
    elif aggr == "sum":
        return torch.sparse_coo_tensor(edge_index,
                                        edge_weight,
                                        size=(n_node, n_node))
    elif aggr == "gcn":
        deg = torch.pow(deg, -0.5)
        return torch.sparse_coo_tensor(edge_index,
                                        deg[edge_index[0]] * edge_weight *
                                        deg[edge_index[1]],
                                        size=(n_node, n_node))
    else:
        raise NotImplementedError


class GLASSConv(torch.nn.Module):
    '''
    A kind of message passing layer we use for GLASS.
    We use different parameters to transform the features of node with different labels individually, and mix them.
    Args:
        aggr: the aggregation method.
        z_ratio: the ratio to mix the transformed features.
    '''
    def __init__(self,
                in_channels: int,
                out_channels: int,
                activation=nn.ReLU(inplace=True),
                aggr="mean",
                z_ratio=0.8,
                dropout=0.2):
        super().__init__()
        self.trans_fns = nn.ModuleList([
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels)
        ])  ## Comb and trans are module lists whose [0] elements are for nodes with label 0, and [1] elements are for nodes with label 1.
        self.comb_fns = nn.ModuleList([
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Linear(in_channels + out_channels, out_channels)
        ])
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)
        self.z_ratio = z_ratio
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        for _ in self.trans_fns:
            _.reset_parameters()
        for _ in self.comb_fns:
            _.reset_parameters()
        self.gn.reset_parameters()

    def forward(self, x_, edge_index, edge_weight, mask):
        # if self.adj.shape[0] == 0:
        #     n_node = x_.shape[0]
        #     self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)
        n_node = x_.shape[0]  ## x_ : input node features (struc + epi + seq).
        self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)
        # transform node features with different parameters individually.
        x1 = self.activation(self.trans_fns[1](x_))
        x0 = self.activation(self.trans_fns[0](x_))  ## trans[0] transforms x_ for nodes with label 0.
        # mix transformed feature.
        ## ## we find that the number of label 0 nodes in much more than that of label 1 nodes, so the modules for label 1 are used less frequently and are not trained well.
        ## ## Therefore, we mix the output of two sets of modules so these modules can be used with the same frequency.
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)  ## z_ratio is a hyperparameter. This trick is wholly based on the observation.
        # pass messages.
        x = self.adj @ x  ## The GLASSConv passes messages.
        x = self.gn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  ## 设置training=self.training,在训练时应用dropout,评估时关闭dropout
        x = torch.cat((x, x_), dim=-1)  ## use an operation similar to residue connection: concatenating the input node feature x_ and the node embedding x. (Lead to higher dimension)
        # transform node features with different parameters individually.
        x1 = self.comb_fns[1](x)
        x0 = self.comb_fns[0](x)  ## use comb[0] to reduce the dimension (comb means combining x_ and x).
        # mix transformed feature.
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0, self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        return x


class EmbZGConv(nn.Module):
    '''
    combination of some GLASSConv layers, normalization layers, dropout layers, and activation function.
    Args:
        max_deg: the max integer in input node features.
        conv: the message passing layer we use.
        gn: whether to use GraphNorm.
        jk: whether to use Jumping Knowledge Network.
    '''
    def __init__(self,
                hidden_channels,
                output_channels,
                num_layers,
                max_deg,
                dropout=0,
                activation=nn.ReLU(),
                conv=GLASSConv,
                gn=True,
                jk=False,
                **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1, hidden_channels, scale_grad_by_freq=False)
        self.emb_gn = GraphNorm(hidden_channels)
        # self.emb_gn = GraphNorm(output_channels)
        self.convs = nn.ModuleList()
        self.jk = jk
        # for _ in range(num_layers - 1):
        #     self.convs.append(
        #         conv(in_channels=hidden_channels,
        #             out_channels=hidden_channels,
        #             activation=activation,
        #              **kwargs))
        # self.convs.append(
        #     conv(in_channels=hidden_channels,
        #         out_channels=output_channels,
        #         activation=activation,
        #          **kwargs))

        self.convs.append(
                conv(in_channels=hidden_channels,
                    out_channels=output_channels,
                    activation=activation,
                     **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                conv(in_channels=output_channels,
                    out_channels=output_channels,
                    activation=activation,
                    **kwargs))

        self.activation = activation
        self.dropout = dropout
        if gn:
            self.gns = nn.ModuleList()
            # for _ in range(num_layers - 1):
            #     self.gns.append(GraphNorm(hidden_channels))
            # if self.jk:
            #     self.gns.append(
            #         GraphNorm(output_channels +
            #                   (num_layers - 1) * hidden_channels))

            for _ in range(num_layers - 1):
                self.gns.append(GraphNorm(output_channels))
            if self.jk:
                self.gns.append(
                    GraphNorm(output_channels +
                              (num_layers - 1) * output_channels))
            else:
                self.gns.append(GraphNorm(output_channels))
        else:
            self.gns = None
        self.reset_parameters()

    def reset_parameters(self):
        self.input_emb.reset_parameters()
        self.emb_gn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, z=None):
        # z is the node label.
        if z is None:
            mask = (torch.zeros(
                (x.shape[0]), device=x.device) < 0.5).reshape(-1, 1)
        else:
            mask = (z > 0.5).reshape(-1, 1)
        # convert integer input to vector node features.
        x = self.input_emb(x).reshape(x.shape[0], -1)
        x = self.emb_gn(x)  ## 这里的输入的x shape： [node_num, dim]
        xs = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        # pass messages at each layer.
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight, mask)
            xs.append(x)
            if not (self.gns is None):
                x = self.gns[layer](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight, mask)
        xs.append(x)

        if self.jk:
            x = torch.cat(xs, dim=-1)
            if not (self.gns is None):
                x = self.gns[-1](x)
            return x
        else:
            x = xs[-1]
            if not (self.gns is None):
                x = self.gns[-1](x)
            return x


class PoolModule(nn.Module):
    '''
    Modules used for pooling node embeddings to produce subgraph embeddings.
    Args:
        trans_fn: module to transfer node embeddings.
        pool_fn: module to pool node embeddings like global_add_pool.
    '''
    def __init__(self, pool_fn, trans_fn=None):
        super().__init__()
        self.pool_fn = pool_fn
        self.trans_fn = trans_fn

    def forward(self, x, batch):
        # The j-th element in batch vector is i if node j is in the i-th subgraph.
        # for example [0,1,0,0,1,1,2,2] means nodes 0,2,3 in subgraph 0, nodes 1,4,5 in subgraph 1, and nodes 6,7 in subgraph 2.
        if self.trans_fn is not None:
            x = self.trans_fn(x)
        return self.pool_fn(x, batch)


class AddPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_add_pool, trans_fn)


class MaxPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_max_pool, trans_fn)


class MeanPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_mean_pool, trans_fn)


class SizePool(AddPool):
    def __init__(self, trans_fn=None):
        super().__init__(trans_fn)

    def forward(self, x, batch):
        if x is not None:
            if self.trans_fn is not None:
                x = self.trans_fn(x)
        x = GraphSizeNorm()(x, batch)
        return self.pool_fn(x, batch)


class GLASS(nn.Module):
    '''
    GLASS model: combine message passing layers and mlps and pooling layers.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''
    def __init__(self, conv: EmbZGConv, preds: nn.ModuleList,
                pools: nn.ModuleList):
        super().__init__()
        self.conv = conv
        self.preds = preds
        self.pools = pools

    def NodeEmb(self, x, edge_index, edge_weight, z=None):
        # shape of x: [node_number, 1, 1]
        embs = []
        for _ in range(x.shape[1]):
            emb = self.conv(x[:, _, :].reshape(x.shape[0], x.shape[-1]),
                            edge_index, edge_weight, z)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)
        return emb
   
    # def Pool(self, emb, subG_node, pool):
    #     batch, pos = pad2batch(subG_node)
    #     emb = emb[pos]
    #     emb = pool(emb, batch)
    #     return emb

    def Pool(self, emb, subG_node, pool):
        batch, pos = pad2batch(subG_node)
        emb = emb[pos]
        emb_size = pool(emb, batch)
        emb_mean = global_mean_pool(emb, batch)
        emb_max  = global_max_pool(emb, batch)
        # emb = torch.cat([emb_mean, emb_max, emb_size], dim=1)

        ### 子图中节点数量及节点间的距离
        batch_size = torch.unique(batch).shape[0]
        node_num_list = []
        node_dis_list = []
        for i in range(batch_size):
            subG = pos[batch == i]
            node_num_list.append(len(subG))
            node_dis_list.append(subG[-1] - subG[0])
        node_num_fea = torch.Tensor(node_num_list).to(emb.device).reshape(-1, 1)
        node_dis_fea = torch.Tensor(node_dis_list).to(emb.device).reshape(-1, 1)
        emb = torch.cat([emb_mean, emb_max, emb_size, node_num_fea, node_dis_fea], dim=1)

        return emb

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0):
        node_emb = self.NodeEmb(x, edge_index, edge_weight, z)  ## 节点embedding
        emb = self.Pool(node_emb, subG_node, self.pools[id])  ## 子图embedding
        emb = self.preds[id](emb)
        return torch.sigmoid(self.preds[1](emb)), node_emb
        # return self.preds[id](emb)

# models used for producing node embeddings (Copy from MATCHA).
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import torch.utils.data as Data
import torch.optim as optim
import time
from tqdm import tqdm, trange
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
device_ids = [0, 1]
activation = torch.tanh


class FeedForward(nn.Module):
	''' A two-feed-forward-layer module '''
	def __init__(self, dims, dropout=None, reshape=False, use_bias=True):
		super(FeedForward, self).__init__()
		self.w_stack = []
		for i in range(len(dims) - 1):
			self.w_stack.append(nn.Linear(dims[i], dims[i + 1], use_bias))
			self.add_module("FF_Linear%d" % (i), self.w_stack[-1])
		
		if dropout is not None:
			self.dropout = nn.Dropout(dropout)
		else:
			self.dropout = None
		
		self.reshape = reshape

	def forward(self, x):
		output = x
		for i in range(len(self.w_stack) - 1):
			output = self.w_stack[i](output)
			output = activation(output)
			if self.dropout is not None:
				output = self.dropout(output)
		output = self.w_stack[-1](output)
		
		if self.reshape:
			output = output.view(output.shape[0], -1, 1)
		
		return output


# Used only for really big adjacency matrix
class SparseEmbedding(nn.Module):
	def __init__(self, embedding_weight, sparse=False):
		super().__init__()
		# print(embedding_weight.shape)  ## adj.shape = (935, 935)
		self.sparse = sparse
		if self.sparse:
			self.embedding = embedding_weight
		else:
			try:
				try:
					self.embedding = torch.from_numpy(
						np.asarray(embedding_weight.todense())).to(device)
				except BaseException:
					self.embedding = torch.from_numpy(
						np.asarray(embedding_weight)).to(device)
			except Exception as e:
				print("Sparse Embedding Error", e)
				self.sparse = True
				self.embedding = embedding_weight
	
	def forward(self, x):
		
		if self.sparse:
			x = x.cpu().numpy()
			x = x.reshape((-1))
			temp = np.asarray((self.embedding[x, :]).todense())
			# temp = np.asarray((self.embedding[x, :]))
			return torch.from_numpy(temp).to(device)
		else:
			return self.embedding[x, :]


class TiedAutoEncoder(nn.Module):
	def __init__(self, shape_list, use_bias=True):
		super().__init__()
		self.weight_list = []
		self.bias_list = []
		self.use_bias = use_bias
		self.recon_bias_list = []
		for i in range(len(shape_list) - 1):
			self.weight_list.append(nn.parameter.Parameter(torch.Tensor(shape_list[i + 1], shape_list[i]).to(device)))
			self.bias_list.append(nn.parameter.Parameter(torch.Tensor(shape_list[i + 1]).to(device)))
			self.recon_bias_list.append(nn.parameter.Parameter(torch.Tensor(shape_list[i]).to(device)))
		self.recon_bias_list = self.recon_bias_list[::-1]
		
		for i, w in enumerate(self.weight_list):
			self.register_parameter('tied weight_%d' % i, w)
			self.register_parameter('tied bias1', self.bias_list[i])
			self.register_parameter('tied bias2', self.recon_bias_list[i])
		
		self.reset_parameters()
	
	def reset_parameters(self):
		for i, w in enumerate(self.weight_list):
			torch.nn.init.kaiming_uniform_(self.weight_list[i], a=math.sqrt(5))
		
		for i, b in enumerate(self.bias_list):
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_list[i])
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.bias_list[i], -bound, bound)
		temp_weight_list = self.weight_list[::-1]
		for i, b in enumerate(self.recon_bias_list):
			fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(temp_weight_list[i])
			bound = 1 / math.sqrt(fan_out)
			torch.nn.init.uniform_(self.recon_bias_list[i], -bound, bound)
	
	def forward(self, input):
		# return input, input
		encoded_feats = input
		for i in range(len(self.weight_list)):
			if self.use_bias:
				encoded_feats = F.linear(encoded_feats, self.weight_list[i], self.bias_list[i])
			else:
				encoded_feats = F.linear(encoded_feats, self.weight_list[i])
			if i < len(self.weight_list) - 1:
				encoded_feats = activation(encoded_feats)
		
		reverse_weight_list = self.weight_list[::-1]
		reconstructed_output = encoded_feats
		for i in range(len(self.recon_bias_list)):
			reconstructed_output = F.linear(reconstructed_output, reverse_weight_list[i].t(), self.recon_bias_list[i])
			if i < len(self.recon_bias_list) - 1:
				reconstructed_output = activation(reconstructed_output)
		
		return encoded_feats, reconstructed_output


class MultipleEmbedding(nn.Module):
    def __init__(
            self,
            embedding_weights,
            dim,
            sparse=True,
            num_list=None,
            chrom_range=None,
            inter_initial=None):
        super().__init__()
        print(dim)   ## 64
        self.num_list = torch.tensor([0] + list(num_list)).to(device)
        print(self.num_list)  ## tensor([  0, 935], device='cuda:0')
        self.chrom_range = chrom_range
        self.dim = dim
        
        self.embeddings = []
        for i, w in enumerate(embedding_weights):
            self.embeddings.append(SparseEmbedding(w, sparse))
        
        if inter_initial is not None:
            for i in trange(len(inter_initial)):
                temp = inter_initial[i, :]
                inter_initial[i, temp > 0] = scipy.stats.mstats.zscore(temp[temp > 0]).astype('float32')
			
			# inter_initial[inter_initial > 0] = scipy.stats.mstats.zscore(inter_initial[inter_initial > 0], axis=1).astype('float32')
            inter_initial[np.isnan(inter_initial)] = 0.0
			
            self.inter_initial = SparseEmbedding(inter_initial, sparse)
        else:
            self.inter_initial = SparseEmbedding(w, sparse)


        test = torch.zeros(1, device=device).long()
        self.input_size = []
        for w in self.embeddings:
            self.input_size.append(w(test).shape[-1])
		
        self.wstack = [TiedAutoEncoder([self.input_size[i], self.dim, self.dim], use_bias=False).to(device) for i, w in enumerate(self.embeddings)]
        self.next_w = FeedForward([self.dim, self.dim]).to(device)
        # self.recon = [FeedForward([self.dim, self.input_size[i]]).to(device) for i, w in enumerate(self.embeddings)]
        self.recon = [FeedForward([self.dim, v[1] - v[0]]).to(device) for i, v in self.chrom_range.items()]
		
        for i, w in enumerate(self.wstack):
            self.add_module("Embedding_Linear%d" % (i), w)
            # self.add_module("Embedding_Linear", self.next_w)
            self.add_module("Embedding_recon%d" % (i), self.recon[i])
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        
        final = torch.zeros((len(x), self.dim)).to(device)
        recon_loss = torch.Tensor([0.0]).to(device)
        for i in range(len(self.num_list) - 1):
            select = (x >= (self.num_list[i] + 1)) & (x < (self.num_list[i + 1] + 1))
            # select = (x >= (self.num_list[i])) & (x < (self.num_list[i + 1]))
            if torch.sum(select) == 0:
                continue
            adj = self.embeddings[i](x[select] - self.num_list[i] - 1)
            output = self.dropout(adj)
            output, recon = self.wstack[i](output)
            # output = self.norm_stack[i](output)
            final[select] = output
        
        final = final

        random_chrom = np.random.choice(np.arange(len(self.chrom_range)), 1)[0]
        # Get the bins in the other chromosome, and it cannot be 0 (because 0 is padding)
        other_chrom = ((x < self.num_list[random_chrom] + 1) | (x >= self.num_list[random_chrom + 1] + 1)) & (x != 0)
        if torch.sum(other_chrom) != 0:
            target = self.inter_initial(x[other_chrom] - 1)
            target = target[:, self.num_list[random_chrom]:self.num_list[random_chrom + 1]]
            recon = self.recon[random_chrom](activation(final[other_chrom]))
            recon_loss += (target - recon).pow(2).mean(dim=-1).mean() * 100

            # recon_loss += sparse_autoencoder_error(recon, adj)

        return final, recon_loss

def sparse_autoencoder_error(y_pred, y_true):
        return (y_true - y_pred).pow(2).mean(dim=-1).mean() * 100
        # return torch.mean(torch.sum((y_true.ne(0).type(torch.float) * (y_true - y_pred)) ** 2, dim = -1) / torch.sum(y_true.ne(0).type(torch.float), dim = -1))



# models used for producing node embeddings (in GLASS paper).

class MyGCNConv(torch.nn.Module):
    '''
    A kind of message passing layer we use for pretrained GNNs.
    Args:
        aggr: the aggregation method.
    '''
    def __init__(self,
                in_channels: int,
                out_channels: int,
                activation=nn.ReLU(inplace=True),
                aggr="mean"):
        super().__init__()
        self.trans_fn = nn.Linear(in_channels, out_channels)
        self.comb_fn = nn.Linear(in_channels + out_channels, out_channels)
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)

    def reset_parameters(self):
        self.trans_fn.reset_parameters()
        self.comb_fn.reset_parameters()
        self.gn.reset_parameters()

    def forward(self, x_, edge_index, edge_weight):
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)
        x = self.trans_fn(x_)
        x = self.activation(x)
        x = self.adj @ x
        x = self.gn(x)
        x = torch.cat((x, x_), dim=-1)
        x = self.comb_fn(x)
        return x


class EmbGConv(torch.nn.Module):
    '''
    combination of some message passing layers, normalization layers, dropout layers, and activation function.
    Args:
        max_deg: the max integer in input node features.
        conv: the message passing layer we use.
        gn: whether to use GraphNorm.
        jk: whether to use Jumping Knowledge Network.
    '''
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 max_deg: int,
                 dropout=0,
                 activation=nn.ReLU(inplace=True),
                 conv=GCNConv,
                 gn=True,
                 jk=False,
                 **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1, hidden_channels)
        self.convs = nn.ModuleList()
        self.jk = jk
        if num_layers > 1:
            self.convs.append(
                conv(in_channels=input_channels,
                     out_channels=hidden_channels,
                     **kwargs))
            for _ in range(num_layers - 2):
                self.convs.append(
                    conv(in_channels=hidden_channels,
                         out_channels=hidden_channels,
                         **kwargs))
            self.convs.append(
                conv(in_channels=hidden_channels,
                     out_channels=output_channels,
                     **kwargs))
        else:
            self.convs.append(
                conv(in_channels=input_channels,
                     out_channels=output_channels,
                     **kwargs))
        self.activation = activation
        self.dropout = dropout
        if gn:
            self.gns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels))
        else:
            self.gns = None
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, z=None):
        xs = []
        x = F.dropout(self.input_emb(x.reshape(-1)),
                      p=self.dropout,
                      training=self.training)
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if not (self.gns is None):
                x = self.gns[layer](x)
            xs.append(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(self.convs[-1](x, edge_index, edge_weight))
        if self.jk:
            return torch.cat(xs, dim=-1)
        else:
            return xs[-1]


class EdgeGNN(nn.Module):
    '''
    EdgeGNN model: combine message passing layers and mlps and pooling layers to do link prediction task.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''
    def __init__(self, conv, preds: nn.ModuleList, pools: nn.ModuleList):
        super().__init__()
        self.conv = conv
        self.preds = preds
        self.pools = pools

    def NodeEmb(self, x, edge_index, edge_weight, z=None):
        embs = []
        for _ in range(x.shape[1]):
            emb = self.conv(x[:, _, :].reshape(x.shape[0], x.shape[-1]),
                            edge_index, edge_weight, z)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)
        return emb

    def Pool(self, emb, subG_node, pool):
        emb = emb[subG_node]
        emb = torch.mean(emb, dim=1)
        return emb

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0):
        emb = self.NodeEmb(x, edge_index, edge_weight, z)
        emb = self.Pool(emb, subG_node, self.pools[id])
        return self.preds[id](emb)
