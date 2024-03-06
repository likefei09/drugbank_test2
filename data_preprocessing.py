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
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
BOND_ORDER_MAP = {0: 0, 1: 1, 1.5: 2, 2: 3, 3: 4}    #0:0未知类型  1：1单键  1.5：2芳香键  2：3双键  3：4三键

#对数据进行预处理，加载数据、生成配对三元组、划分数据集

#将分子对象的特征信息添加到特征字典集合中
def set_feat_dict(mol, feat_dicts_raw):
    for atom in mol.GetAtoms():
        feat_dicts_raw[0].add(atom.GetSymbol())   #原子符号
        feat_dicts_raw[1].add(atom.GetDegree())   #原子度数
        feat_dicts_raw[2].add(atom.GetImplicitValence())    #隐式价电子数
        feat_dicts_raw[3].add(atom.GetFormalCharge())       #形式电荷
        feat_dicts_raw[4].add(atom.GetNumRadicalElectrons())    #自由基电子数
        feat_dicts_raw[5].add(int(atom.GetHybridization()))     #杂化化合物类型
        feat_dicts_raw[6].add(atom.GetTotalNumHs())             #氢原子数
        feat_dicts_raw[7].add(int(atom.GetIsAromatic()))        #是否是芳香原子


#从分子对象中提取节点特征，每个原子都被标识为一个特征向量
def get_node_feat(mol, feat_dicts_raw):
    atom_idx_to_node_idx = {}
    x = [[] for i in range(8)]

    #把每一个分子中的每一个原子的feat_dict值对应的放到上面的x列表中的对应位置
    for i, atom in enumerate(mol.GetAtoms()):
        atom_idx_to_node_idx[atom.GetIdx()] = i

        x[0].append(feat_dicts_raw[0][atom.GetSymbol()])
        x[1].append(feat_dicts_raw[1][atom.GetDegree()])
        x[2].append(feat_dicts_raw[2][atom.GetImplicitValence()])
        x[3].append(feat_dicts_raw[3][atom.GetFormalCharge()])
        x[4].append(feat_dicts_raw[4][atom.GetNumRadicalElectrons()])
        x[5].append(feat_dicts_raw[5][int(atom.GetHybridization())])
        x[6].append(feat_dicts_raw[6][atom.GetTotalNumHs()])
        x[7].append(feat_dicts_raw[7][int(atom.GetIsAromatic())])

    feat_dim = [len(feat) for feat in feat_dicts_raw]   #feat_dim: [38(38个原子),7,4,7,5,7,5,2]  不知道为什么放在这里，每个原子都算一遍？

    for i in range(8):
        cur = torch.LongTensor(x[i])
        cur = F.one_hot(cur, feat_dim[i])
        x[i] = cur

    x = torch.cat(x, dim=-1)
    x = x.float()

    #x是生成的该分子的每个原子的特征的one_hot编码， atom_idx_to_node_idx是这个分子一共有多少原子，类如第一个分子有17个原子，返回的就是{0:0,1:1,2:2,3:3……,16：16}
    return x, atom_idx_to_node_idx


#确定分子中原子之间的连接关系，并且生成边索引的张量cur_edge_index
def get_edge_index(mol, atom_idx_to_node_idx):
    cur_edge_index = []
    #对于分子中的每个键，先找起始原子和终点原子，然后把这两个之间的边（正反两条），加在cur_edge_index中
    for bond in mol.GetBonds():
        atom_1 = bond.GetBeginAtomIdx()
        atom_2 = bond.GetEndAtomIdx()

        node_1 = atom_idx_to_node_idx[atom_1]
        node_2 = atom_idx_to_node_idx[atom_2]

        cur_edge_index.append([node_1, node_2])
        cur_edge_index.append([node_2, node_1])

    if len(cur_edge_index) > 0:
        cur_edge_index = torch.LongTensor(cur_edge_index).t()
    else:
        cur_edge_index = torch.LongTensor(2, 0)

    return cur_edge_index

#将特征值转换成字典形式，并将处理后的特征字典添加到列表feat_dicts中
def proc_feat_dicts( feat_dicts_raw):
    dict_names = ["symbol", "deg", "valence", "charge", "electron", "hybrid", "hydrogen", "aromatic"]
    feat_dicts = []

    output_lines = ["{\n"]

    for feat_dict, dict_name in zip(feat_dicts_raw, dict_names):
        feat_dict = sorted(list(feat_dict))
        feat_dict = {item: i for i, item in enumerate(feat_dict)}
        feat_dicts.append(feat_dict)

    return feat_dicts

##############################################################################################
def get_atoms_info(mol):
    atoms = mol.GetAtoms()
    n_atom = len(atoms)
    atom_fea = torch.zeros(7, n_atom, dtype=torch.half)
    AllChem.ComputeGasteigerCharges(mol)
    for idx, atom in enumerate(atoms):
        atom_fea[0, idx] = atom.GetAtomicNum()  #原子序数
        atom_fea[1, idx] = atom.GetTotalDegree() + 1    #原子总度数+1
        atom_fea[2, idx] = int(atom.GetHybridization()) + 1     #原子的杂化方式
        atom_fea[3, idx] = atom.GetTotalNumHs() + 1     #愿你在的氢原子数+1
        atom_fea[4, idx] = atom.GetIsAromatic() + 1  #原子是否为芳香原子
        for n_ring in range(3, 9):
            if atom.IsInRingSize(n_ring):
                atom_fea[5, idx] = n_ring + 1
                break
        else:
            if atom.IsInRing():
                atom_fea[5, idx] = 10
        atom_fea[6, idx] = atom.GetDoubleProp("_GasteigerCharge")*10

    atom_fea = torch.nan_to_num(atom_fea)
    return np.array(atom_fea), n_atom

#处理图边的信息部分
def get_bond_order_adj(mol):    #没用到
    n_atom = len(mol.GetAtoms())
    bond_adj = torch.zeros(n_atom, n_atom, dtype=torch.uint8)

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_adj[i, j] = bond_adj[j, i] = BOND_ORDER_MAP[bond.GetBondTypeAsDouble()]
    return np.array(bond_adj)
def get_bond_adj(mol):
    """
    :param mol: rdkit mol
    :return: multi graph for {
                sigmoid_bond_graph,
                pi_bond_graph,
                2pi_bond_graph,
                aromic_graph,
                conjugate_graph,
                ring_graph,
    }
    """
    n_atom = len(mol.GetAtoms())
    bond_adj = torch.zeros(n_atom, n_atom, dtype=torch.uint8)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        bond_adj[i, j] = bond_adj[j, i] = 1
        if bond_type in [2, 3]:
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 1)
        if bond_type == 3:
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 2)
        if bond_type == 1.5:
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 3)
        if bond.GetIsConjugated():
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 4)
        if bond.IsInRing():
            bond_adj[i, j] = bond_adj[j, i] = bond_adj[i, j] + (1 << 5)
    return np.array(bond_adj)
#处理边的距离部分
def get_dist_adj(mol, use_3d_info=False):
    if use_3d_info:
        m2 = Chem.AddHs(mol)
        is_success = AllChem.EmbedMolecule(m2, enforceChirality=False)
        if is_success == -1:
            dist_adj = None
        else:
            AllChem.MMFFOptimizeMolecule(m2)
            m2 = Chem.RemoveHs(m2)
            dist_adj = (-1 * torch.tensor(AllChem.Get3DDistanceMatrix(m2), dtype=torch.float))
    else:
        dist_adj = (-1 * torch.tensor(molDG.GetMoleculeBoundsMatrix(mol), dtype=torch.float))
    return np.array(dist_adj)
#总的调用函数
def smile_to_mol_info(smile, calc_dist=True, use_3d_info=False):
    mol = Chem.MolFromSmiles(smile)
    bond_adj = get_bond_adj(mol)  #边信息
    dist_adj = get_dist_adj(mol) if calc_dist else None   #距离信息
    dist_adj_3d = get_dist_adj(mol, use_3d_info) if calc_dist else None
    atom_fea, n_atom = get_atoms_info(mol)   #分子信息，总共多少个分子
    return atom_fea,bond_adj,dist_adj,n_atom
atoms,adjs,dists,nums=[],[],[],[]
##############################################################################################


#生成药物数据的图数据图像
def generate_drug_data(mol, feat_dicts, calc_dist=True,use_3d_info=False):
    cur_x, atom_idx_to_node_idx = get_node_feat(mol, feat_dicts)
    cur_edge_index = get_edge_index(mol, atom_idx_to_node_idx)

    cur_bond_adj = get_bond_adj(mol)  # 边信息
    cur_dist_adj = get_dist_adj(mol) if calc_dist else None  # 距离信息
    cur_dist_adj_3d = get_dist_adj(mol, use_3d_info) if calc_dist else None
    cur_atom_fea, n_atom = get_atoms_info(mol)  # 分子信息，总共多少个分子


    return Data(x=cur_x, edge_index=cur_edge_index,bond_adj=cur_bond_adj,dist_adj=cur_dist_adj,atom_fea=cur_atom_fea)


#加载药物分子数据并生成对应的图数据对象
def load_drug_mol_data(args):

    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    drug_id_mol_tup = []
    drug_smile_dict = {}   #smile：Simplified molecular input line entry system 简化分子线性输入规范

    feat_dicts_raw = [set() for i in range(8)]    #特征字典（存储所有的分子的所有的特征） 创建了一个包含8个空集合的列表

    for id1, id2, smiles1, smiles2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_s1], data[args.c_s2], data[args.c_y]):
        drug_smile_dict[id1] = smiles1
        drug_smile_dict[id2] = smiles2
    
    for id, smiles in drug_smile_dict.items():
        mol =  Chem.MolFromSmiles(smiles.strip())
        set_feat_dict(mol, feat_dicts_raw)
        if mol is not None:
            drug_id_mol_tup.append((id, mol))

    feat_dicts = proc_feat_dicts(feat_dicts_raw)


    drug_data = {id: generate_drug_data(mol, feat_dicts) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs')}
    save_data(drug_data, 'drug_data.pkl', args)
    return drug_data

#生成正负样本的三元组对象
def generate_pair_triplets(args):
    #print()
    pos_triplets = []   #正三元组
    drug_ids = []  #所有药物的id

    with open(f'{args.dirname}/{args.dataset.lower()}/drug_data.pkl', 'rb') as f:
        drug_ids = list(pickle.load(f).keys())
        # print(drug_ids)

    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    for id1, id2, relation in zip(data[args.c_id1], data[args.c_id2],  data[args.c_y]):
        if ((id1 not in drug_ids) or (id2 not in drug_ids)): continue
        # Drugbank dataset is 1-based index, need to substract by 1
        if args.dataset in ('drugbank', ):
            relation -= 1
        pos_triplets.append([id1, id2, relation])

    if len(pos_triplets) == 0:
        raise ValueError('All tuples are invalid.')

    pos_triplets = np.array(pos_triplets)
    data_statistics = load_data_statistics(pos_triplets)
    drug_ids = np.array(drug_ids)

    neg_samples = []   #负样本
    for pos_item in tqdm(pos_triplets, desc='Generating Negative sample'):
        temp_neg = []
        h, t, r = pos_item[:3]   #id1 id2 relation 切片处理，提取前三个元素

        if args.dataset == 'drugbank':
            neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics, drug_ids, args)
            temp_neg = [str(neg_h) + '$h' for neg_h in neg_heads] + \
                        [str(neg_t) + '$t' for neg_t in neg_tails]
        else:
            existing_drug_ids = np.asarray(list(set(
                np.concatenate([data_statistics["ALL_TRUE_T_WITH_HR"][(h, r)], data_statistics["ALL_TRUE_H_WITH_TR"][(h, r)]], axis=0)
                )))
            temp_neg = _corrupt_ent(existing_drug_ids, args.neg_ent, drug_ids, args)
        
        neg_samples.append('_'.join(map(str, temp_neg[:args.neg_ent])))
    
    df = pd.DataFrame({'Drug1_ID': pos_triplets[:, 0], 
                        'Drug2_ID': pos_triplets[:, 1], 
                        'Y': pos_triplets[:, 2],
                        'Neg samples': neg_samples})
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    df.to_csv(filename, index=False)
    print(f'\nData saved as {filename}!')
    save_data(data_statistics, 'data_statistics.pkl', args)


#计算用于生成负样本的概率，并存储其它与数据相关的统计信息
def load_data_statistics(all_tuples):
    '''
    This function is used to calculate the probability in order to generate a negative. 
    You can skip it because it is unimportant.
    '''
    print('Loading data statistics ...')
    statistics = dict()
    statistics["ALL_TRUE_H_WITH_TR"] = defaultdict(list)
    statistics["ALL_TRUE_T_WITH_HR"] = defaultdict(list)
    statistics["FREQ_REL"] = defaultdict(int)
    statistics["ALL_H_WITH_R"] = defaultdict(dict)
    statistics["ALL_T_WITH_R"] = defaultdict(dict)
    statistics["ALL_TAIL_PER_HEAD"] = {}
    statistics["ALL_HEAD_PER_TAIL"] = {}

    for h, t, r in tqdm(all_tuples, desc='Getting data statistics'):
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)].append(h)
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)].append(t)
        statistics["FREQ_REL"][r] += 1.0
        statistics["ALL_H_WITH_R"][r][h] = 1
        statistics["ALL_T_WITH_R"][r][t] = 1

    for t, r in statistics["ALL_TRUE_H_WITH_TR"]:
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)] = np.array(list(set(statistics["ALL_TRUE_H_WITH_TR"][(t, r)])))
    for h, r in statistics["ALL_TRUE_T_WITH_HR"]:
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)] = np.array(list(set(statistics["ALL_TRUE_T_WITH_HR"][(h, r)])))

    for r in statistics["FREQ_REL"]:
        statistics["ALL_H_WITH_R"][r] = np.array(list(statistics["ALL_H_WITH_R"][r].keys()))
        statistics["ALL_T_WITH_R"][r] = np.array(list(statistics["ALL_T_WITH_R"][r].keys()))
        statistics["ALL_HEAD_PER_TAIL"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_T_WITH_R"][r])
        statistics["ALL_TAIL_PER_HEAD"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_H_WITH_R"][r])
    
    print('getting data statistics done!')

    return statistics

#生成一定数量的破坏实体，用于负样本的生成或其他应用场景。它确保生成的破坏实体与已存在的正样本实体和之前生成的破坏实体不重复。
def _corrupt_ent(positive_existing_ents, max_num, drug_ids, args):
    corrupted_ents = []
    while len(corrupted_ents) < max_num:
        candidates = args.random_num_gen.choice(drug_ids, (max_num - len(corrupted_ents)) * 2, replace=False)  #从drug_ids 中选两个元素作为候选项
        invalid_drug_ids = np.concatenate([positive_existing_ents, corrupted_ents], axis=0)  #将已存在的正样本实体 positive_existing_ents 和之前生成的破坏实体列表 corrupted_ents 进行拼接，生成一个无效药物标识符的数组
        mask = np.isin(candidates, invalid_drug_ids, assume_unique=True, invert=True)
        corrupted_ents.extend(candidates[mask])

    corrupted_ents = np.array(corrupted_ents)[:max_num]
    return corrupted_ents

#根据给定的正样本和数据统计信息生成一组正常的批次数据，包括正样本和对应数量的负样本
def _normal_batch( h, t, r, neg_size, data_statistics, drug_ids, args):  #neg_size=1
    neg_size_h = 0
    neg_size_t = 0
    prob = data_statistics["ALL_TAIL_PER_HEAD"][r] / (data_statistics["ALL_TAIL_PER_HEAD"][r] + 
                                                            data_statistics["ALL_HEAD_PER_TAIL"][r])
    # prob = 2     ----->错的
    for i in range(neg_size):
        if args.random_num_gen.random() < prob:   #如果随机生成的数小于prob
            neg_size_h += 1
        else:
            neg_size_t +=1
    
    return (_corrupt_ent(data_statistics["ALL_TRUE_H_WITH_TR"][t, r], neg_size_h, drug_ids, args),
            _corrupt_ent(data_statistics["ALL_TRUE_T_WITH_HR"][h, r], neg_size_t, drug_ids, args))  

#保存数据到文件中
def save_data(data, filename, args):
    dirname = f'{args.dirname}/{args.dataset}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + '/' + filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')

#代码将数据集拆分为训练集和测试集，并保存为多个折
def split_data(args):
    test_size_ratio = args.test_ratio   #从args中获取测试集大小比例
    n_folds = args.n_folds    #折数fold
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    df = pd.read_csv(filename)
    seed = args.seed
    class_name = args.class_name
    save_to_filename = os.path.splitext(filename)[0]
    cv_split = StratifiedShuffleSplit(n_splits=n_folds, test_size=test_size_ratio, random_state=seed)  #构造一个分层随机划分的交叉验证拆分器，指定折数，测试集大小比例，和随机数种子
    for fold_i, (train_index, test_index) in enumerate(cv_split.split(X=df, y=df[class_name])):
        print(f'Fold {fold_i} generated!')
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        train_df.to_csv(f'{save_to_filename}_train_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_train_fold{fold_i}.csv', 'saved!')
        test_df.to_csv(f'{save_to_filename}_test_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_test_fold{fold_i}.csv', 'saved!')



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #向解释器添加命令行参数
    parser.add_argument('-d', '--dataset', type=str,  choices=['drugbank', 'twosides'], default='drugbank',
                            help='Dataset to preprocess.')
    parser.add_argument('-n', '--neg_ent', type=int, default=1, help='Number of negative samples')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('-o', '--operation', type=str,  choices=['all', 'generate_triplets', 'drug_data', 'split'], default='all', help='Operation to perform')   #操作
    parser.add_argument('-t_r', '--test_ratio', type=float, default=0.2)   #测试集比例
    parser.add_argument('-n_f', '--n_folds', type=int, default=3)   #几折

    #定义了一个字典对象，包含两个键值对，每个键值对表示一个数据集和对应的列名
    dataset_columns_map = {
        'drugbank': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'twosides': ('Drug1_ID', 'Drug2_ID', 'Drug1', 'Drug2', 'New Y'),
    }

    #定义了一个字典，包含两个键值对 数据集名称和对应的文件名 分隔符
    dataset_file_name_map = {
        'drugbank': ('data/drugbank.tab', '\t'),
        'twosides': ('data/twosides_ge_500.csv', ',')
    }
    #解析命令行参数
    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    args.c_id1, args.c_id2, args.c_s1, args.c_s2, args.c_y = dataset_columns_map[args.dataset]
    args.dataset_filename, args.delimiter = dataset_file_name_map[args.dataset]    #文件名和分隔符
    args.dirname = 'data/preprocessed'

    args.random_num_gen = np.random.RandomState(args.seed)
    if args.operation in ('all', 'drug_data'):
        load_drug_mol_data(args)

    if args.operation in ('all','generate_triplets'):
        generate_pair_triplets(args)
    
    if args.operation in ('all', 'split'):
        args.class_name = 'Y'
        split_data(args)
