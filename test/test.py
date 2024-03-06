from operator import index
import torch
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import torch.nn.functional as F

from rdkit import Chem

if __name__ == '__main__':
    bsz = 1
    n = 4
    d = 3
    a = torch.randint(10,(bsz, n, d))
    b = torch.randint(10,(bsz, 5, d))
    print('a=', a.shape)
    print('b=', b.shape)
    outer = torch.einsum("...bc,...de->...bdce", a, b)
    print(outer.shape)
    a = a.view(bsz, n, 1, d, 1)
    b = b.view(bsz, 1, 5, 1, d)
    outer1 = a * b
    print(outer1.shape)
    outer1 = outer1.view(outer1.shape[:-2] + (-1,))
    print(outer1.shape)



# h_atom_fea =

# class Trainer(object):
#     def __init__(self, model, lr, weight_decay, batch):
#         self.model = model
#         # w - L2 regularization ; b - not L2 regularization
#         weight_p, bias_p = [], []
#
#         for p in self.model.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#         for name, p in self.model.named_parameters():
#             if 'bias' in name:
#                 bias_p += [p]
#             else:
#                 weight_p += [p]
#         # self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
#         self.optimizer_inner = RAdam(
#             [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
#         self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
#         self.batch = batch
#
#     def train(self, dataset, device):
#         self.model.train()
#         np.random.shuffle(dataset)
#         N = len(dataset)
#         loss_total = 0
#         i = 0
#         self.optimizer.optimizer.zero_grad()
#         dists,adjs,atoms,proteins,labels = [],[],[],[],[]
#         for data in dataset:
#             i = i+1
#             atom, adj,dist,protein,label = data
#             adjs.append(adj)
#             atoms.append(atom)
#             dists.append(dist)
#             proteins.append(protein)
#             labels.append(label)
#             if i % self.batch == 0 or i == N:
#                 data_pack = pack(atoms,adjs,dists,proteins,labels,device)
#                 loss = self.model(data_pack)
#                 loss = loss / self.batch
#                 loss.backward()
#                 # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
#                 dists,adjs, atoms, proteins, labels = [], [], [],[],[]
#             else:
#                 continue
#             if i % self.batch == 0 or i == N:
#                 self.optimizer.step()
#                 self.optimizer.zero_grad()
#             loss_total += loss.item()
#         return loss_total
#
# def pack(atoms,adjs,adj_dist,proteins,labels, device):
#     atoms_len = 0
#     proteins_len = 0
#     N = len(atoms)
#     atom_num = []
#     for atom in atoms:
#         atom_num.append(atom.shape[1])
#         if atom.shape[1] >= atoms_len:
#             atoms_len = atom.shape[1]
#     protein_num = []
#
#     for protein in proteins:
#         protein_num.append(protein.shape[0])
#         if protein.shape[0] >= proteins_len:
#             proteins_len = protein.shape[0]
#     '''
#     #print(proteins.shape)
#     for protein in proteins:
#         protein_num.append(len(protein))
#         if len(protein) >= proteins_len:
#             proteins_len = len(protein)
#     '''
#     #atoms_new = torch.zeros((N,7,atom_num), device=device)
#     atoms_new = torch.zeros((N, 7, atoms_len), device=device)
#     i = 0
#     for atom in atoms:
#         a_len = atom.shape[1]
#         atoms_new[i, :, :a_len] = atom
#         i += 1
#     #print(atoms_new[0])
#
#     i = 0
#     for adj in adjs:
#         a_len = adj.shape[0]
#         adj=adj.cuda()
#         #print(adj.is_cuda)
#         adjs_new = torch.zeros((N, a_len, a_len), device=device)
#         #print(adjs_new.is_cuda)
#         adj = adj + torch.eye(a_len, device=device)
#         adjs_new[i, :a_len, :a_len] = adj
#         i += 1
#
#
#     i = 0
#     for dist in adj_dist:
#         a_len = dist.shape[0]
#         dist=dist.cuda()
#         adj_dist_new = torch.zeros((N, a_len, a_len), device=device)  # size:atoms_len*atoms_len
#         dist = dist + torch.eye(a_len, device=device)
#         adj_dist_new[i, :a_len, :a_len] = dist
#         i += 1
#
#     proteins_new = torch.zeros((N, proteins_len, 100), device=device)
#     i = 0
#     for protein in proteins:
#         a_len = protein.shape[0]
#         #protein=torch.from_numpy(protein)
#         proteins_new[i, :a_len, :] = protein
#         i += 1
#     '''
#
#     proteins_new = torch.zeros((N, proteins_len), device=device)
#     i = 0
#     for protein in proteins:
#         a_len = len(protein)
#         # print(protein)
#         proteins_new[i, :a_len] = protein
#         i += 1
#     '''
#     labels_new = torch.zeros(N, dtype=torch.long, device=device)
#     i = 0
#     for label in labels:
#         labels_new[i] = label
#         i += 1
#     return (atoms_new, adjs_new,adj_dist_new,proteins_new, labels_new, atom_num, protein_num)
#
# def load_tensor(file_name, dtype):
#     result=[]
#     np.load.__defaults__=(None, True, True, 'ASCII')
#     for d in np.load(file_name):
#         #print(d)
#         #temp=dtype(d).to(device)
#         result.append(d)
#     np.load.__defaults__=(None, False, True, 'ASCII')
#     return result
#
# # 指定.npy文件的路径
# file_path = '/mnt/sdb/home/hjy/exp1/dataset/Davis/bond_adj.npy'
# bond_adj = load_tensor(file_path, torch.FloatTensor)
# adjs = bond_adj[8:12]
#
# i = 0
# for adj in adjs:
#     a_len = adj.shape[0]
#     print(a_len)
#     adj = adj
#     # print(adj.is_cuda)
#     adjs_new = torch.zeros((8, a_len, a_len))
#     # print(adjs_new.is_cuda)
#     adj = adj + torch.eye(a_len)
#     adjs_new[i, :a_len, :a_len] = adj
#     i += 1
# print(adjs_new.size())
# print(bond_adj[2].shape)

# 使用numpy.load()函数加载.npy文件
# data = np.load(file_path, allow_pickle=True)
# print(data)
# print(data[2].shape)


# 读取 pickle 文件
# with open('/mnt/sdb/home/lkf/code/drugbank_test2/data/preprocessed/drugbank/drug_data.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(data)
# with open('/mnt/sdb/home/lkf/code/drugbank/data/preprocessed/drugbank/data_statistics.pkl', 'rb') as f:
#     data = pickle.load(f)
#     # print(data)
#     prob = data["ALL_TAIL_PER_HEAD"]['1'] / (data["ALL_TAIL_PER_HEAD"]['1'] +
#                                                             data["ALL_HEAD_PER_TAIL"]['1'])
#     print(prob)

# filename = '/mnt/sdb/home/lkf/code/drugbank/data/preprocessed/drugbank/data_statistics.pkl'
# file = open(filename, 'rb')
# data = pickle.load(file)
# print(data)



# def set_feat_dict(mol, feat_dicts_raw):
#     for atom in mol.GetAtoms():
#         feat_dicts_raw[0].add(atom.GetSymbol())   #原子符号
#         feat_dicts_raw[1].add(atom.GetDegree())   #原子度数
#         feat_dicts_raw[2].add(atom.GetImplicitValence())    #隐式价电子数
#         feat_dicts_raw[3].add(atom.GetFormalCharge())       #形式电荷
#         feat_dicts_raw[4].add(atom.GetNumRadicalElectrons())    #自由基电子数
#         feat_dicts_raw[5].add(int(atom.GetHybridization()))     #杂化化合物类型
#         feat_dicts_raw[6].add(atom.GetTotalNumHs())             #氢原子数
#         feat_dicts_raw[7].add(int(atom.GetIsAromatic()))        #是否是芳香原子
#
#
# #从分子对象中提取节点特征，每个原子都被标识为一个特征向量
# # def get_node_feat(mol, feat_dicts_raw):
# #     atom_idx_to_node_idx = {}
# #     x = [[] for i in range(8)]
# #
# #     for i, atom in enumerate(mol.GetAtoms()):
# #         atom_idx_to_node_idx[atom.GetIdx()] = i
# #
# #         x[0].append(feat_dicts_raw[0][atom.GetSymbol()])
# #         x[1].append(feat_dicts_raw[1][atom.GetDegree()])
# #         x[2].append(feat_dicts_raw[2][atom.GetImplicitValence()])
# #         x[3].append(feat_dicts_raw[3][atom.GetFormalCharge()])
# #         x[4].append(feat_dicts_raw[4][atom.GetNumRadicalElectrons()])
# #         x[5].append(feat_dicts_raw[5][int(atom.GetHybridization())])
# #         x[6].append(feat_dicts_raw[6][atom.GetTotalNumHs()])
# #         x[7].append(feat_dicts_raw[7][int(atom.GetIsAromatic())])
# #
# #     feat_dim = [len(feat) for feat in feat_dicts_raw]
# #
# #     for i in range(8):
# #         cur = torch.LongTensor(x[i])
# #         cur = F.one_hot(cur, feat_dim[i])
# #         x[i] = cur
# #
# #     x = torch.cat(x, dim=-1)
# #     x = x.float()
# #
# #     return x, atom_idx_to_node_idx
# #从分子对象中提取节点特征，每个原子都被标识为一个特征向量
# def get_node_feat(mol, feat_dicts_raw):
#     atom_idx_to_node_idx = {}
#     x = [[] for i in range(8)]
#
#     for i, atom in enumerate(mol.GetAtoms()):
#         atom_idx_to_node_idx[atom.GetIdx()] = i
#
#         x[0].append(feat_dicts_raw[0][atom.GetSymbol()])
#         # print("x[0]=",x[0])
#         x[1].append(feat_dicts_raw[1][atom.GetDegree()])
#         # print("x[1]",x[1])
#         x[2].append(feat_dicts_raw[2][atom.GetImplicitValence()])
#         x[3].append(feat_dicts_raw[3][atom.GetFormalCharge()])
#         x[4].append(feat_dicts_raw[4][atom.GetNumRadicalElectrons()])
#         x[5].append(feat_dicts_raw[5][int(atom.GetHybridization())])
#         x[6].append(feat_dicts_raw[6][atom.GetTotalNumHs()])
#         x[7].append(feat_dicts_raw[7][int(atom.GetIsAromatic())])
#
#     feat_dim = [len(feat) for feat in feat_dicts_raw]
#     # print("x=",x)
#     for i in range(8):
#         cur = torch.LongTensor(x[i])
#         # print(cur.shape)
#         cur = F.one_hot(cur, feat_dim[i])
#         x[i] = cur
#
#     # print(x)
#     x = torch.cat(x, dim=-1)
#     x = x.float()
#     # print('x=', x)
#     # print('x.shape=', x.shape)
#     print('atom_idx_to_node_idx=', atom_idx_to_node_idx)
#     return x, atom_idx_to_node_idx
#
#
# #确定分子中原子之间的连接关系，并且生成边索引的张量cur_edge_index
# def get_edge_index(mol, atom_idx_to_node_idx):
#     cur_edge_index = []
#
#     for bond in mol.GetBonds():
#         # print('bond=', bond)
#         atom_1 = bond.GetBeginAtomIdx()
#         atom_2 = bond.GetEndAtomIdx()
#         # print('atom_1=',atom_1)
#         # print('atom_2=',atom_2)
#
#         node_1 = atom_idx_to_node_idx[atom_1]
#         node_2 = atom_idx_to_node_idx[atom_2]
#
#         cur_edge_index.append([node_1, node_2])
#         cur_edge_index.append([node_2, node_1])
#
#     if len(cur_edge_index) > 0:
#         cur_edge_index = torch.LongTensor(cur_edge_index).t()
#     else:
#         cur_edge_index = torch.LongTensor(2, 0)
#     print(cur_edge_index)
#     return cur_edge_index
#
# def proc_feat_dicts( feat_dicts_raw):
#     dict_names = ["symbol", "deg", "valence", "charge", "electron", "hybrid", "hydrogen", "aromatic"]
#     feat_dicts = []
#
#     output_lines = ["{\n"]
#
#     for feat_dict, dict_name in zip(feat_dicts_raw, dict_names):
#         # print('feat_dict=', feat_dict)
#         # print('dict_name=', dict_name)
#         feat_dict = sorted(list(feat_dict))
#         feat_dict = {item: i for i, item in enumerate(feat_dict)}
#         feat_dicts.append(feat_dict)
#
#     return feat_dicts
#
# #生成药物数据的图数据图像
# def generate_drug_data(mol, feat_dicts):
#     cur_x, atom_idx_to_node_idx = get_node_feat(mol, feat_dicts)
#     cur_edge_index = get_edge_index(mol, atom_idx_to_node_idx)
#
#     return Data(x=cur_x, edge_index=cur_edge_index)
#
# drug_id_mol_tup = []
# drug_smile_dict = {'DB04571': 'CC1=CC2=CC3=C(OC(=O)C=C3C)C(C)=C2O1',
#                    'DB00855': 'NCC(=O)CCC(O)=O',
#                    'DB09536': 'O=[Ti]=O',
#                    }
#
# feat_dicts_raw = [set() for i in range(8)]
# print(feat_dicts_raw)
# for id, smiles in drug_smile_dict.items():
#     mol = Chem.MolFromSmiles(smiles.strip())
#     print(len(mol.GetAtoms()))
#     # print(mol.GetPropNames())
#     # for item in mol:
#     #     print(item, end=' ')
#     # atoms = mol.GetAtoms()
#     # for atom in atoms:
#     #     print(atom.GetSymbol())
#     # for atom_idx, atom in enumerate(mol.GetAtoms()):
#     #     print(f'{atom_idx}  {atom.GetSymbol()}')
#
#     set_feat_dict(mol, feat_dicts_raw)
#     print('feat_dicts_raw=', feat_dicts_raw)
#     if mol is not None:
#         drug_id_mol_tup.append((id, mol))
# print(drug_id_mol_tup)
# print('feat_dicts_raw=', feat_dicts_raw)
# # dict_names = ["symbol", "deg", "valence", "charge", "electron", "hybrid", "hydrogen", "aromatic"]
# # print(zip(feat_dicts_raw, dict_names))
#
# feat_dicts = proc_feat_dicts(feat_dicts_raw)
# print('feat_dicts=', feat_dicts)
#
# drug_data = {id: generate_drug_data(mol, feat_dicts) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs')}
# print(drug_data)

