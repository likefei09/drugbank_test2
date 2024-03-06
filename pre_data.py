import os

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
#from tape import TAPETokenizer
from rdkit import Chem
import torch
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
BOND_ORDER_MAP = {0: 0, 1: 1, 1.5: 2, 2: 3, 3: 4}    #0:0未知类型  1：1单键  1.5：2芳香键  2：3双键  3：4三键
# dataset=pd.read_csv(r'G:\序列预测相关\模型\graph-transformer\dataset\KIBA\KIBA.txt',sep = ' ',
#                   names =['drug id','protein id','drug','protein','compound'])



#处理protein部分
# protein=dataset['protein']
# def seq_to_kmers(seq, k=3):
#     """ Divide a string into a list of kmers strings.
#
#     Parameters:
#         seq (string)
#         k (int), default 3
#     Returns:
#         List containing a list of kmers.
#     """
#     N = len(seq)
#     return [seq[i:i+k] for i in range(N - k + 1)]
# class Corpus(object):
#     """ An iteratable for training seq2vec models. """
#
#     def __init__(self,data, ngram):
#         self.df = data
#         self.ngram = ngram
#
#     def __iter__(self):
#         for no, data in enumerate(self.df):
#             yield  seq_to_kmers(data,self.ngram)
# def get_protein_embedding(model,protein):
#     """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
#     vec = np.zeros((len(protein), 100))
#     i = 0
#     for word in protein:
#         vec[i, ] = model.wv[word]
#         i += 1
#     vec=torch.from_numpy(vec)
#     return np.array(vec)
# sent_corpus = Corpus(protein,3)
# model = Word2Vec(vector_size=100, window=5, min_count=0, workers=6)
# model.build_vocab(sent_corpus)
# model.train(sent_corpus,epochs=30,total_examples=model.corpus_count)
# model.save("word2vec_30_DrugBank.model")
# #model=Word2Vec.load('word2vec_30_DrugBank.model')
# print('model end!')
# #model = Word2Vec.load('/mnt/sdb/home/hjy/exp1/word2vec_30_GPCR_train.model')
# proteins=[]
# for no, data in enumerate(protein):
#     protein_embedding = get_protein_embedding(model, seq_to_kmers(data))
#     proteins.append(protein_embedding)
# print('protein end')

#处理drug部分（把其转换为分子图）
# drug=dataset['drug']
# drug= drug.values.tolist()
#处理分子信息部分
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
# for no, data in enumerate(drug):
#     fea,adj,dist,atom_num=smile_to_mol_info(data)
#     atoms.append(fea)
#     adjs.append(adj)
#     dists.append(dist)
#     nums.append(atom_num)
# print('drug end')
# #atoms= torch.tensor([item.cpu().detach().numpy() for item in atoms]).cuda()
# interactions=dataset['compound'].values.tolist()
# dir_input = 'G:/序列预测相关/模型/DrugormerDTI/dataset/DrugBank/'
# os.makedirs(dir_input, exist_ok=True)
# np.save(dir_input + 'atom_feat',atoms )
# np.save(dir_input + 'bond_adj', adjs)
# np.save(dir_input + 'dist_adj',dists)
# np.save(dir_input + 'proteins', proteins)
# np.save(dir_input + 'interactions', interactions)
#
# print('The preprocess of dataset has finished!')

# %%
if __name__ == "__main__":
    batch_mol_list = ["C1=NC2=CC(C34C[NH2+]CC3C4)=CC=C2S1", "O=S1CC2=CC(N3CCOCC3)=CC=C2N1", "C1CCC(C2=[NH+]CCCN2)CC1",
                      "C1=COC2OCCC2C1", "N=C1CCN=N1", "C1=CC=C2CN3CCNCCNCCCCNCCNCC4CC5CCCCC5N4CC3CC2=C1",
                      "C1=CC2=CC(C3=NNC4=CC=C(C5=NN=C(C6CCC[NH2+]C6)O5)C=C43)=CC=C2C1",
                      "C1=CC=C(C2=CC=C3NC4=C(C=CCC4)C3=C2)C=C1",
                      # "O=S1C2=CC=CC=C2C2=C3C1=C(C1=C4C=CC(=C(C5=CC6=C7C8=CC=CC=C8S(=O)C5=C7S(=O)C5=CC=CC=C56)C5=CC=C(N5)C(C5=CC6=C7C8=CC=CC=C8S(=O)C5=C7S(=O)C5=CC=CC=C56)=C5C=CC(=C(C6=CC7=C8C9=CC=CC=C9S(=O)C6=C8S(=O)C6=CC=CC=C67)C6=CC=C1N6)N5)N4)",
                      "C1=CC2=CC(=C1)COC1CCC3C(CCC4C5CCC(CCCCNN=CC6=CC=C(C=C6)C[NH2+]CCNN=C2)C5CCC34)C1",
                      "C1=CC(C2=NN3C=CC(C4=C[NH+]=C(C5CC6CC6N5)N4)=CC3=C2)=CC=C1C1=C[NH+]=C(C2CC3CC3N2)N1",
                      "C1CCC(C2(C3(C4(C5(C6(C7(C8(C9(C%10(C%11(C%12(C%13(C%14(C%15CCCCC%15)CCCCC%14)CCCCC%13)CCCCC%12)CCCCC%11)CCCCC%10)CCCCC9)CCCCC8)CCCCC7)CCCCC6)CCCCC5)CCCCC4)CCCCC3)CCCCC2)CC1",
                      "C1=CC2=C(S1)C1=S=C(C3=CC=C(C4=CC5=C(C=C6C=C(C7=CC=C(C8=CC9=C(S8)C8=C(C=CS8)N9)C8=NSN=C78)SC6=C5)S4)C4=NSN=C34)C=C1N2",
                      "C1=CC=C2C(=C1)C1=CC=CC=C1N2C1=CC(C2=CC(C3=CC=C4CC[N+]5=CC=CC=C5C4=C3)=CC=C2)=CC(N2C3=CC=CC=C3C3=CC=CC=C32)=C1",
                      "C1=CC=C(C2=C3C=CC(=N3)C(C3=CC=CC=C3)=C3C=CC(=C(C4=CC=CC=C4)C4=NC(=C(C5=CC=CC=C5)C5=CC=C2N5)C2=C4CC4CC(C2)C2=C4C4=C(C5=CC=CC=C5)C5=CC=C(N5)C(C5=CC=CC=C5)=C5C=CC(=N5)C(C5=CC=CC=C5)=C5C=CC(=C(C6=CC=CC=C6)C2=N4)N5)N3)C=C1",
                      ]

    for index, smiles in enumerate(batch_mol_list):
        mol = AllChem.MolFromSmiles(smiles)
        atom_fea,bond_adj,dist_adj,n_atom = smile_to_mol_info(smiles)
        print(len(smiles))
        print(bond_adj)
        # print(mol)
        # data = get_atoms_info(mol)
        # data = get_bond_order_adj(mol)
        # data = get_bond_adj(mol)
        # data = get_dist_adj(mol)
        # print(data)
    #    pickle.dump(data, open(f"./sampleData/{index}.pkl", "wb"))
    print("finished")

