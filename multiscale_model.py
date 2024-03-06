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
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
import torch_geometric.nn as gnn
from torch.nn.init import xavier_uniform_, constant_
from torch import Tensor
from collections import OrderedDict
import numpy as np
import copy
import math
PI = 3.14159
A = (2 * PI) ** 0.5
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


# class DrugEncoder(torch.nn.Module):
#     def __init__(self, in_dim,  hidden_dim):
#         super().__init__()
#
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.PReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.PReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#         )
#         self.lin0 = nn.Linear(in_dim, hidden_dim)
#         self.line_graph =  GraphDenseNet(num_input_features=hidden_dim, out_dim=hidden_dim, block_config=[8, 8, 8], bn_sizes=[2, 2, 2])
#
#     def forward(self, data):
#         data.x = self.mlp(data.x)
#         x = self.line_graph(data)
#
#         return x

#不知道要干什么，反正需要进行参数的初始化就对了
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class GraphormerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu,
                 layer_norm_eps=1e-5, device='cuda',batch_first=False, norm_first=False) -> None:
        super(GraphormerEncoderLayer, self).__init__()
        self.device=device
        self.self_attn = MultiHeadAtomAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,device=device)

        self.linear1 = nn.Linear(d_model, dim_feedforward).cuda()
        self.dropout = nn.Dropout(dropout).cuda()
        self.linear2 = nn.Linear(dim_feedforward, d_model).cuda()

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps).cuda()
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps).cuda()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super(GraphormerEncoderLayer, self).__setstate__(state)

    def forward(self, x, attn_mask=None):
        #print((self.norm1(x + self._sa_block(x, attn_mask))).is_cuda)
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class MultiHeadAtomAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,device='cuda',add_zero_attn=False,
                 batch_first=False) -> None:
        super(MultiHeadAtomAttention, self).__init__()
        self.device=device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)).to(device))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim).to(device))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias).cuda()

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        q_d, bsz, _ = q.size()
        k_d = k.size(0)
        q = q.contiguous().view(q_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # if attn_mask is not None and attn_mask.size(0) != bsz * self.num_heads:
        #     attn_mask = attn_mask.reshape(bsz, 1, q_d, k_d) \
        #         .expand(bsz, self.num_heads, q_d, k_d).reshape(bsz * self.num_heads, q_d, k_d)

        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)

        dropout_p = self.dropout if self.training else 0.0
        attn_output, attn_output_weights = \
            _scaled_dot_product_atom_attention(q, k, v, dropout_p, attn_mask)

        attn_output = attn_output.transpose(0, 1).contiguous().view(q_d, bsz, self.embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, q_d, k_d)
            return attn_output, attn_output_weights.permute(0, 2, 3, 1).contiguous()
        else:
            return attn_output, None
def _scaled_dot_product_atom_attention(q, k, v=None, dropout_p=0.0, attn_mask=None):
    """
    :param attn_mask:
    :param q: [bsz, q, d]
    :param k: [bsz, k, d]
    :param v: [bsz, k, d]
    :param dropout_p: p in [0, 1]
    :return:([bsz, q, d], [bsz, q, k]) or (None, [bsz, q, k]) if v is None
    """
    B, Q, D = q.size()
    # print("q.size:", q.size(), k.size())
    q = q / math.sqrt(D)
    attn = torch.bmm(q, k.transpose(-2, -1))
    #这里更改了，感觉要是要对齐要大改
    #if attn_mask is not None:
    #    attn += attn_mask
        # attn = torch.nan_to_num(attn)
    if v is not None:
        attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    output = torch.bmm(attn, v) if v is not None else None
    # print(f"q:{q}\n"
    #       f"k:{k}\n"
    #       f"v:{v}\n"
    #       f"attn_mask:{attn_mask}"
    #       f"attn:{attn}")
    # raise ValueError
    return output, attn
def _in_projection_packed(q, k, v=None, w=None, b=None):
    """
    :param q: [q, bsz, d]
    :param k: [k, bsz, d]
    :param v: [v, bsz, d]
    :param w: [d*3, d]
    :param b: [d*3]
    :return: projected [q, k, v] or [q, k] if v is None
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    elif v is not None:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    elif q is k:
        return F.linear(q, w, b).chunk(2, dim=-1)
    else:
        w_q, w_k = w.split([E, E])
        if b is None:
            b_q = b_k = None
        else:
            b_q, b_k = b.split([E, E])
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k)

class MultiHeadAtomAdj(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, device='cuda', add_zero_attn=False,
                 batch_first=False) -> None:
        super(MultiHeadAtomAdj, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.device = device
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj_weight = nn.Parameter(torch.empty((2 * embed_dim, embed_dim)).to(device))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(2 * embed_dim).to(device))
        else:
            self.register_parameter('in_proj_bias', None)
        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)

    def forward(self, query, key, attn_mask=None):
        if self.batch_first:
            query, key = [x.transpose(1, 0) for x in (query, key)]
        q, k = _in_projection_packed(query, key, None, self.in_proj_weight, self.in_proj_bias)
        q_d, bsz, _ = q.size()
        k_d = k.size(0)
        q = q.contiguous().view(q_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # if attn_mask is not None and attn_mask.size(0) != bsz * self.num_heads:
        #     attn_mask = attn_mask.reshape(bsz, 1, q_d, k_d) \
        #         .expand(bsz, self.num_heads, q_d, k_d).reshape(bsz * self.num_heads, q_d, k_d)

        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)

        _, adj = _scaled_dot_product_atom_attention(q, k, None, 0.0, attn_mask)

        return adj.view(bsz, self.num_heads, q_d, k_d).permute(0, 2, 3, 1).contiguous()


# 5.整个drug处理部分形成一个统一的encoder,直接掉用他放在predict里！

#以下是整个获取embedding的方法！！
class GraphormerEmbeddingLayer(nn.Module):
    def __init__(self, d_model, n_head, max_paths, n_graph_type, max_single_hop, atom_dim, total_degree,
                 hybrid, hydrogen, aromatic, ring, n_layers, need_graph_token,
                 use_3d_info=False, dropout=0.,use_dist_adj=True):
        super(GraphormerEmbeddingLayer, self).__init__()

        self.atom_encoder = AtomFeaEmbedding(d_model, atom_dim, total_degree, hybrid,
                                             hydrogen, aromatic, ring, n_layers, need_graph_token)
        self.edge_encoder = EdgeEmbedding(d_model, n_head, max_paths, n_graph_type, max_single_hop, n_layers,
                                          need_graph_token, use_3d_info=use_3d_info, use_dist_adj=use_dist_adj)

    def forward(self, atom_fea, bond_adj, dist_adj,dist3d_adj=None, contrast=False):
        return self.atom_encoder(atom_fea,contrast), \
               self.edge_encoder(bond_adj, dist_adj, dist3d_adj, contrast)

#根据数据集的处理，目前知道的有总原子数，总建数，杂化方式，含氢量，是否为芳香烃，几元环等特征，所以init对应其one-hot需要的维度，不够可以调整
class AtomFeaEmbedding(nn.Module):
    def __init__(self, d_model, atom_dim=100, total_degree=10, hybrid=10,
                 hydrogen=8, aromatic=2, ring=10, n_layers=1, need_graph_token=True):
        super(AtomFeaEmbedding, self).__init__()
        self.atom_encoders = nn.ModuleList([
            nn.Embedding(100, d_model, padding_idx=0).cuda(),
            nn.Embedding(total_degree + 1, d_model, padding_idx=0).cuda(),
            nn.Embedding(20, d_model, padding_idx=0).cuda(),
            nn.Embedding(hydrogen + 1, d_model, padding_idx=0).cuda(),
            nn.Embedding(aromatic + 1, d_model, padding_idx=0).cuda(),
            nn.Embedding(ring + 1, d_model, padding_idx=0).cuda(),
            GaussianAtomLayer(d_model, means=(-1, 1), stds=(0.1, 10))
        ])
        self.need_graph_token = need_graph_token
        # self.feed_forward = nn.Sequential(
        #     nn.LayerNorm(d_model),
        # )

        if need_graph_token:
            self.graph_token = nn.Embedding(2, d_model).cuda()
            self.contrast_token = nn.Embedding(2, d_model).cuda()

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self,atom_fea,contrast=False):
        """
        :param atom_fea: [bsz, n_fea_type, n_atom]
        :return: [bsz, n_atom + I, d_model] I = 1 if need_graph_token else 0
        """
        #这里改过，atom_fea格式不一样

        bsz, n_fea_type, n_atom = atom_fea.size()
        #print(atom_fea)
        #print('atom_fea embeddings:',atom_fea.shape)
        #print('n_atom',n_atom)
        out = self.atom_encoders[-1](atom_fea[:, -1])
        #print(self.atom_encoders[2](atom_fea[:, 2].int()).is_cuda)
        #print(self.embedding.num_embeddings)
        for idx in range(6):
            #print(atom_fea[:, idx].int())
            #print(self.atom_encoders[idx].weight.size(), atom_fea[:, idx].max(dim=0))
            #print(out.is_cuda)
            out += self.atom_encoders[idx](atom_fea[:, idx].int())
            #out += self.atom_encoders[idx](atom_fea[:, idx])
            #print(out)
        if self.need_graph_token:
            graph_token = self.graph_token.weight[0].clone()
            if contrast:
                graph_token += self.contrast_token.weight[0]
            graph_token = graph_token.view(1, 1, -1).repeat(bsz, 1, 1)
            out = torch.cat([graph_token, out], dim=1)
        return out
class GaussianAtomLayer(nn.Module):
    def __init__(self, d_model=128, means=(0, 3), stds=(0.1, 10),device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.means = nn.Embedding(1, d_model).cuda()
        self.stds = nn.Embedding(1, d_model).cuda()
        self.mul = nn.Embedding(1, 1).cuda()
        self.bias = nn.Embedding(1, 1).cuda()
        nn.init.uniform_(self.means.weight, means[0], means[1])
        nn.init.uniform_(self.stds.weight, stds[0], stds[1])
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)
    def real_forward(self, x):
        mul = self.mul.weight
        bias = self.bias.weight
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, self.d_model)
        mean = self.means.weight[0]
        std = self.stds.weight[0].abs() + 1e-5
        return gaussian(x, mean, std).type_as(self.means.weight)
    def forward(self, x):
        """
        :param x: [bsz, n_atom]
        :return: [bsz, n_atom, d_model]
        """
        out = self.real_forward(x)
        return torch.where(x.unsqueeze(-1).expand_as(out) != 0, out, torch.zeros_like(out))
#2.Spatial Encoding（空间编码）的方法
#对应edge_embedding层中spatial encoder的部分，运用了Gaussian核函数处理方法(可以尝试替换为其他核函数)
def gaussian(x, mean, std):
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (A * std)
class GaussianBondLayer(nn.Module):
    def __init__(self, nhead=16, means=(0, 3), stds=(0.1, 10)):
        super().__init__()
        self.nhead = nhead
        self.means = nn.Embedding(1, nhead).cuda()
        self.stds = nn.Embedding(1, nhead).cuda()
        self.mul = nn.Embedding(1, 1).cuda()
        self.bias = nn.Embedding(1, 1).cuda()
        nn.init.uniform_(self.means.weight, means[0], means[1])
        nn.init.uniform_(self.stds.weight, stds[0], stds[1])
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def real_forward(self, x):
        mul = self.mul.weight
        bias = self.bias.weight
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.nhead)
        mean = self.means.weight[0]
        std = self.stds.weight[0].abs() + 1e-5
        return gaussian(x, mean, std).type_as(self.means.weight)

    def forward(self, x):
        """
        :param x: [bsz, n_atom, n_atom]
        :return: [bsz, n_atom, n_atom, nhead]
        """
        out = self.real_forward(x)
        return torch.where(x.unsqueeze(-1).expand_as(out) != 0, out, torch.zeros_like(out))



class DrugEncoder(nn.Module):
    def __init__(self,d_model=512,nhead=32,num_layer=8,dim_feedforward=512,dropout=0.1,atom_dim=92,total_degree=10,hybrid=6,hydrogen=9,aromatic=2,ring=10,n_layers=1,
                 max_paths=50, n_graph_type=6, max_single_hop=4,
                 activation=F.gelu,use_3d_info=False,use_dist_adj=True,need_graph_token=True,batch_first=True,norm_first=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layer = num_layer
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.use_3d_info = use_3d_info
        self.use_dist_adj = use_dist_adj
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.n_graph_type = n_graph_type
        self.max_single_hop = max_single_hop
        self.max_paths=max_paths
        encoder_layer = GraphormerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                batch_first=batch_first, norm_first=norm_first)
        self.emb = GraphormerEmbeddingLayer(d_model, nhead, max_paths, n_graph_type, max_single_hop, atom_dim, total_degree,
                 hybrid, hydrogen, aromatic, ring, n_layers, need_graph_token=True, use_3d_info=use_3d_info,
                                      dropout=dropout,use_dist_adj=use_dist_adj)
        self.encoder = GraphormerEncoder(encoder_layer, num_layer, nn.LayerNorm(d_model))
    def forward(self,atom_fea,bond_adj,dist_adj):
        """{
            atom_fea:[bsz, n_type, n_atom]
            bond_adj: [bsz, n_atom, n_atom]
            dist_adj:
            center_cnt:
            rxn_type:
            dist3d_adj:
            lg_dic:
        }
        :return:
        """

        bsz, n_atom, _ = bond_adj.size()
        atom_fea, masked_adj = self.emb(atom_fea, bond_adj, dist_adj)
        fea = self.encoder(atom_fea, masked_adj)
        #fea = self.encoder(shared_atom_fea, masked_adj)
        #print('drug embedding:',fea.shape)
        return fea

#3.Edge Encoding(边编码)的方法
#对应edge_emebedding层中edge encoder的部分
#因为在模型当中edge_encoding和spatial_encoding最后要concat在一起以后喂给transformer,所以这里EdgeEmbedding的输出是把两个结合起来过以后！
class EdgeEmbedding(nn.Module):
    def __init__(self, embed_dim, n_head=16, max_paths=50, n_graph_type=6, max_single_hop=4, n_layers=1,
                 need_graph_token=True, use_3d_info=False, use_dist_adj=True):
        """
        :param embed_dim:
        :param n_head:  n_head mast has to be 2 to the nth power
                        example: n_head=2^4, hop_distribution=[1, 2^3, 2^2, 2^1, 2^0]
                                 n_head=2^n, hop_distribution=[1, 2^(n-1), 2^(n-2), ..., 2^0]
        :param max_paths:
        :param n_graph_type:
        :param max_single_hop:
        :param n_layers:
        """
        super().__init__()
        self.num_heads = n_head
        self.embed_dim = embed_dim
        # assert n_head & (n_head - 1) == 0, f"n_head mast has to be 2 to the nth power, but got{n_head}"
        self.n_graph_type = n_graph_type
        self.max_single_hop = max_single_hop
        self.use_3d_info = use_3d_info
        self.use_dist_adj = use_dist_adj
        # self.hop_distribution = [1 << i for i in range(int(math.log2(n_head) - 1), -1, -1)]
        # self.hop_distribution[-1] += 1
        # assert sum(self.hop_distribution) == self.num_heads

        self.head_dim = embed_dim // n_head
        assert self.head_dim * n_head == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.edge_encoders = nn.ModuleList([
            nn.Embedding(max_paths + 1, n_head, padding_idx=0).cuda() for _ in range(n_graph_type)
        ])
        self.spatial_encoder = GaussianBondLayer(n_head, means=(0, 3), stds=(0.1, 10))
        if self.use_3d_info:
            self.spatial3d_encoder = GaussianBondLayer(n_head, means=(0, 3), stds=(0.1, 10))

        # self.norm = nn.LayerNorm(n_head)

        self.need_graph_token = need_graph_token
        if need_graph_token:
            self.graph_token = nn.Embedding(1, n_head).cuda()
            self.contrast_token = nn.Embedding(1, n_head).cuda()

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, bond_adj, dist_adj, dist3d_adj=None, contrast=False):
        """
        :param bond_adj: [bsz, n_atom, n_atom]
        :param dist_adj: [bsz, n_atom, n_atom]
        :return: attention bias with mask [bsz*n_head, n_atom, n_atom]
        """
        # bsz, n_hop, n_type, n_atom, _ = bond_adj.size()
        # bond_embed = self.edge_encoders[0](bond_adj[:, :, 0].int()).sum(dim=1)
        # for i in range(1, self.n_graph_type):
        #     bond_embed += self.edge_encoders[i](bond_adj[:, :, i].int()).sum(dim=1)

        # [bsz, n_hop, n_type, n_atom, n_atom, n_head] -> # [bsz, n_atom, n_atom, n_head]

        # bond_embed = torch.cat([
        #     bond_embed[:, i].unsqueeze(1).expand(bsz, hop, n_atom, n_atom)
        #     for i, hop in enumerate(self.hop_distribution)
        # ], dim=1)  # [bsz, n_hop, n_type, n_atom, n_atom] -> # [bsz, n_head, n_atom, n_atom]
        bsz, n_atom, _ = bond_adj.size()
        #print('n_atom:',n_atom)
        comb_embed = torch.zeros(bsz, n_atom, n_atom, self.num_heads, device=bond_adj.device)
        if self.use_dist_adj and dist_adj is not None:
            comb_embed += self.spatial_encoder(dist_adj)
        if self.use_3d_info and dist3d_adj is not None:
            comb_embed += self.spatial3d_encoder(dist3d_adj)


        if self.max_single_hop > 0:
            for i in range(self.n_graph_type):
                j_hop_embed = bond_adj.long()
                # decode to multi sense embedding
                j_hop_embed = torch.where(j_hop_embed > 0, ((j_hop_embed - 1) >> i) % 2, 0).float()
                base_hop_embed = j_hop_embed
                comb_embed += self.edge_encoders[i](j_hop_embed.int())
                for j in range(1, self.max_single_hop):
                    # generate multi atom environment embedding
                    j_hop_embed = torch.bmm(j_hop_embed, base_hop_embed)
                    comb_embed += self.edge_encoders[i](j_hop_embed.int())

        comb_embed = comb_embed.permute(0, 3, 1, 2)
        mask = torch.where(bond_adj != 0, 0., -torch.inf)

        if self.need_graph_token:
            graph_token = self.graph_token.weight[0].clone()
            if contrast:
                graph_token += self.contrast_token.weight[0]
            graph_token = graph_token.view(1, self.num_heads, 1, 1).repeat(bsz, 1, 1, 1)

            comb_embed = torch.cat([graph_token.expand(bsz, self.num_heads, n_atom, 1),
                                    comb_embed], dim=-1)
            comb_embed = torch.cat([graph_token.expand(bsz, self.num_heads, 1, n_atom + 1),
                                    comb_embed], dim=-2)

            mask = torch.cat([torch.zeros_like(mask[:, :, 0]).unsqueeze(-1), mask], dim=-1)
            mask = torch.cat([torch.zeros_like(mask[:, 0, :]).unsqueeze(-2), mask], dim=-2)

            # [bsz, n_head, n_atom+1, n_atom+1]
            n_atom += 1
        mask = mask.unsqueeze(1).expand(bsz, self.num_heads, n_atom, n_atom)
        out=(comb_embed + mask).reshape(bsz * self.num_heads, n_atom, n_atom)
        return out

#4.graphormer的模型主题部分
class GraphormerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(GraphormerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm.cuda()

    def forward(self, x, attn_mask=None):
        output = x
        for mod in self.layers:
            output = mod(output, attn_mask)

        if self.norm is not None and self.num_layers > 0:
            #print(output.is_cuda)
            #print(self.norm(output).is_cuda)
            output = self.norm(output)
        return output
class GraphormerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu,
                 layer_norm_eps=1e-5, device='cuda',batch_first=False, norm_first=False) -> None:
        super(GraphormerEncoderLayer, self).__init__()
        self.device=device
        self.self_attn = MultiHeadAtomAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,device=device)

        self.linear1 = nn.Linear(d_model, dim_feedforward).cuda()
        self.dropout = nn.Dropout(dropout).cuda()
        self.linear2 = nn.Linear(dim_feedforward, d_model).cuda()

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps).cuda()
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps).cuda()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super(GraphormerEncoderLayer, self).__setstate__(state)

    def forward(self, x, attn_mask=None):
        #print((self.norm1(x + self._sa_block(x, attn_mask))).is_cuda)
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
class MultiHeadAtomAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,device='cuda',add_zero_attn=False,
                 batch_first=False) -> None:
        super(MultiHeadAtomAttention, self).__init__()
        self.device=device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)).to(device))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim).to(device))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias).cuda()

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        q_d, bsz, _ = q.size()
        k_d = k.size(0)
        q = q.contiguous().view(q_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # if attn_mask is not None and attn_mask.size(0) != bsz * self.num_heads:
        #     attn_mask = attn_mask.reshape(bsz, 1, q_d, k_d) \
        #         .expand(bsz, self.num_heads, q_d, k_d).reshape(bsz * self.num_heads, q_d, k_d)

        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)

        dropout_p = self.dropout if self.training else 0.0
        attn_output, attn_output_weights = \
            _scaled_dot_product_atom_attention(q, k, v, dropout_p, attn_mask)

        attn_output = attn_output.transpose(0, 1).contiguous().view(q_d, bsz, self.embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, q_d, k_d)
            return attn_output, attn_output_weights.permute(0, 2, 3, 1).contiguous()
        else:
            return attn_output, None
def _scaled_dot_product_atom_attention(q, k, v=None, dropout_p=0.0, attn_mask=None):
    """
    :param attn_mask:
    :param q: [bsz, q, d]
    :param k: [bsz, k, d]
    :param v: [bsz, k, d]
    :param dropout_p: p in [0, 1]
    :return:([bsz, q, d], [bsz, q, k]) or (None, [bsz, q, k]) if v is None
    """
    B, Q, D = q.size()
    # print("q.size:", q.size(), k.size())
    q = q / math.sqrt(D)
    attn = torch.bmm(q, k.transpose(-2, -1))
    #这里更改了，感觉要是要对齐要大改
    #if attn_mask is not None:
    #    attn += attn_mask
        # attn = torch.nan_to_num(attn)
    if v is not None:
        attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    output = torch.bmm(attn, v) if v is not None else None
    # print(f"q:{q}\n"
    #       f"k:{k}\n"
    #       f"v:{v}\n"
    #       f"attn_mask:{attn_mask}"
    #       f"attn:{attn}")
    # raise ValueError
    return output, attn
def _in_projection_packed(q, k, v=None, w=None, b=None):
    """
    :param q: [q, bsz, d]
    :param k: [k, bsz, d]
    :param v: [v, bsz, d]
    :param w: [d*3, d]
    :param b: [d*3]
    :return: projected [q, k, v] or [q, k] if v is None
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    elif v is not None:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    elif q is k:
        return F.linear(q, w, b).chunk(2, dim=-1)
    else:
        w_q, w_k = w.split([E, E])
        if b is None:
            b_q = b_k = None
        else:
            b_q, b_k = b.split([E, E])
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k)

def pack(atoms, adjs, adj_dist, device):
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[1])
        if atom.shape[1] >= atoms_len:
            atoms_len = atom.shape[1]
    #atoms_new = torch.zeros((N,7,atom_num), device=device)
    atoms_new = torch.zeros((N, 7, atoms_len), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[1]
        atoms_new[i, :, :a_len] = torch.tensor(atom)
        i += 1
    #print(atoms_new[0])

    i = 0
    for adj in adjs:
        adj = torch.tensor(adj)
        a_len = adj.shape[0]
        adj=adj.to(device)
        #print(adj.is_cuda)
        adjs_new = torch.zeros((N, a_len, a_len), device=device)
        #print(adjs_new.is_cuda)
        adj = adj + torch.eye(a_len, device=device)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1


    i = 0
    for dist in adj_dist:
        dist = torch.tensor(dist)
        a_len = dist.shape[0]
        dist=dist.to(device)
        adj_dist_new = torch.zeros((N, a_len, a_len), device=device)  # size:atoms_len*atoms_len
        dist = dist + torch.eye(a_len, device=device)
        adj_dist_new[i, :a_len, :a_len] = dist
        i += 1

    return (atoms_new, adjs_new,adj_dist_new)

class My_DDI(torch.nn.Module):
    def __init__(self, in_dim,  hidden_dim=32*3):
        super(My_DDI, self).__init__()
        print('My_DDI')

        self.drug_encoder = DrugEncoder()

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

    def forward(self, triples, device):
        h_data, t_data, rels, _ = triples
        print('h_data=', h_data.x.shape)
        h_atom_fea, h_bond_adj, h_dist_adj = h_data['atom_fea'], h_data['bond_adj'], h_data['dist_adj']
        t_atom_fea, t_bond_adj, t_dist_adj = t_data['atom_fea'], t_data['bond_adj'], t_data['dist_adj']
        h_atom_fea, h_bond_adj, h_dist_adj = pack(h_atom_fea, h_bond_adj, h_dist_adj, device)
        t_atom_fea, t_bond_adj, t_dist_adj = pack(t_atom_fea, t_bond_adj, t_dist_adj, device)
        print('h_atom_fea=', h_atom_fea.shape)
        # h_bond_adj = torch.tensor(h_bond_adj)
        # h_atom_fea, h_bond_adj, h_dist_adj = torch.tensor(h_atom_fea), torch.tensor(h_bond_adj), torch.tensor(h_dist_adj)
        # h_bond_adj = h_data['bond_adj']
        # h_dist_adj = h_data['dist_adj']
        x_h = self.drug_encoder(h_atom_fea, h_bond_adj, h_dist_adj)
        print('x_h=', x_h.shape)
        x_t = self.drug_encoder(t_atom_fea, t_bond_adj, t_dist_adj)
        # x_h = self.drug_encoder(h_data)
        # x_t = self.drug_encoder(t_data)


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

