import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import GraphConv,GCNConv
from torch_geometric.nn.models import GAE
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import degree, to_dense_batch
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict
import numpy as np
from torch_geometric.nn import  SAGPooling
from torch_geometric.nn import global_max_pool , global_mean_pool,global_add_pool





class NodeLevelBatchNorm(_BatchNorm):
    r"""
    Applies Batch Normalization over a batch of graph data.
    Shape:
        - Input: [batch_nodes_dim, node_feature_dim]
        - Output: [batch_nodes_dim, node_feature_dim]
    batch_nodes_dim: all nodes of a batch graph
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)

class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = F.relu(self.norm(self.conv(x, edge_index)))

        return data

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)


    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features

        data.x = torch.cat(data.x, 1)

        return data


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features

        data = self.conv1(data)

        return data

    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]

        data = self.bn_function(data)
        data = self.conv2(data)

        return data

class GraphDenseNet(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config = (3, 3, 3, 3), bn_sizes=[2, 3, 4, 4]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i+1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, num_input_features // 2)
            self.features.add_module("transition%d" % (i+1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        data = self.features(data)
        x = self.classifer(data.x)

        return x


class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = 6 * n_feats
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )
        self.lin5 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, n_feats)
        )

    def forward(self, x):
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2
        x = self.lin5(x)

        return x


class DrugEncoder(torch.nn.Module):
    def __init__(self, in_dim,  hidden_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.lin0 = nn.Linear(in_dim, hidden_dim)
        self.line_graph =  GraphDenseNet(num_input_features=hidden_dim, out_dim=hidden_dim, block_config=[8, 8, 8], bn_sizes=[2, 2, 2])

    def forward(self, data):
        data.x = self.mlp(data.x)
        x = self.line_graph(data)

        return x


class My_DDI(torch.nn.Module):
    def __init__(self, in_dim,  hidden_dim=32*3):
        super(My_DDI, self).__init__()
        print('My_DDI')

        self.drug_encoder = DrugEncoder(in_dim, hidden_dim)

        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.rmodule = nn.Embedding(86, hidden_dim)

        self.w_j = nn.Linear(hidden_dim, hidden_dim)
        self.w_i = nn.Linear(hidden_dim, hidden_dim)

        self.prj_j = nn.Linear(hidden_dim, hidden_dim)
        self.prj_i = nn.Linear(hidden_dim, hidden_dim)
        self.pool = Attention(dim=hidden_dim)

    def forward(self, triples):
        h_data, t_data, rels, _ = triples

        x_h = self.drug_encoder(h_data)
        x_t = self.drug_encoder(t_data)


        x_h, mask_h = to_dense_batch(x_h, h_data.batch)
        x_t, mask_t = to_dense_batch(x_t, t_data.batch)
        mask_h = (mask_h == False)
        mask_t = (mask_t == False)
        h_final, t_final = self.pool(x_h, x_t, [mask_h, mask_t])
        pair = torch.cat([h_final, t_final], dim=-1)
        rfeat = self.rmodule(rels)
        logit = (self.lin(pair) * rfeat).sum(-1)


        return logit

#子结构交互学习模块：
class Attention(nn.Module):
    def __init__(self,  dim, num_heads = 4):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads
        self.linear_q = nn.Linear(dim, self.dim_per_head*num_heads)
        self.linear_k = nn.Linear(dim, self.dim_per_head*num_heads)
        self.linear_v = nn.Linear(dim, self.dim_per_head*num_heads)
        #self.norm = nn.BatchNorm1d(dim)
        self.norm = nn.LayerNorm(dim)
        self.linear_final = nn.Linear(dim,dim)

        self.linear_q_inner = nn.Linear(dim, dim)
        self.linear_k_inner = nn.Linear(dim, dim)
        self.linear_v_inner = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(p=0.2)




    def attention(self, q1,k1,v1,q2,k2,v2,attn_mask=None,flag=False):
        #print('k1',k1[0])
        #print(True in torch.isnan(k1[0]))
        #print('q1',q2[0])
        #print(True in torch.isnan(q1[0]))

        a1 = torch.tanh(torch.bmm(k1, q2.transpose(1, 2)))
        a2 = torch.tanh(torch.bmm(k2, q1.transpose(1, 2)))
        #print(a1[0])
        if attn_mask is not None:
            #a1=a1.masked_fill(attn_mask, -np.inf)
            #a2=a2.masked_fill(attn_mask.transpose(1, -1), -np.inf)
            mask1 = attn_mask[0]
            mask2 = attn_mask[1]

            a1 = torch.softmax(torch.sum(a1, dim=2).masked_fill(mask1, -np.inf), dim=-1).unsqueeze(dim=1)
            a2 = torch.softmax(torch.sum(a2, dim=2).masked_fill(mask2, -np.inf), dim=-1).unsqueeze(dim=1)
        else:
            a1 = torch.softmax(torch.sum(a1, dim=2), dim=1).unsqueeze(dim=1)
            a2 = torch.softmax(torch.sum(a2, dim=2), dim=1).unsqueeze(dim=1)
            #print('after softmax',a1[0])

        a1 = self.dropout(a1)
        a2 = self.dropout(a2)
        #print(a1.shape, v1.shape, mask1.shape)

        vector1 = torch.bmm(a1, v1).squeeze()
        vector2 = torch.bmm(a2, v2).squeeze()

        return vector1,vector2

    def forward(self, fingerprint_vectors1,  fingerprint_vectors2, attn_mask=None, flag=False):
        #batch_size = fingerprint_vectors1.shape[0]
        #fingerprint_vectors1 = self.self_attention(fingerprint_vectors1)
        #fingerprint_vectors2 = self.self_attention(fingerprint_vectors2)


        q1, q2 = torch.relu(self.linear_q(fingerprint_vectors1)), torch.relu(self.linear_q(fingerprint_vectors2))
        k1, k2 = torch.relu(self.linear_k(fingerprint_vectors1)), torch.relu(self.linear_k(fingerprint_vectors2))
        v1, v2 = torch.relu(self.linear_v(fingerprint_vectors1)), torch.relu(self.linear_v(fingerprint_vectors2))
        '''
        q1, q2 = fingerprint_vectors1,fingerprint_vectors2
        k1, k2 = fingerprint_vectors1, fingerprint_vectors2
        v1, v2 = fingerprint_vectors1, fingerprint_vectors2
        '''
        vector1, vector2 = self.attention(q1,k1,v1,q2,k2,v2, attn_mask, flag)


        vector1 = self.norm(torch.mean(fingerprint_vectors1, dim=1) + vector1)
        vector2 = self.norm(torch.mean(fingerprint_vectors2, dim=1) + vector2)


        return vector1, vector2

