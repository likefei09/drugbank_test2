import os
import torch
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import numpy as np

# %%  两部分组成的？？
class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

def get_bipartite_graph(graph_1,graph_2):
    x1 = np.arange(0,graph_1.num_nodes)
    x2 = np.arange(0,graph_2.num_nodes)
    edge_list = torch.LongTensor(np.meshgrid(x1,x2))
    edge_list = torch.stack([edge_list[0].reshape(-1),edge_list[1].reshape(-1)])
    return edge_list

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

class DrugDataset(Dataset):
    def __init__(self, data_df, drug_graph): #self是自己的实例对象
        self.data_df = data_df
        self.drug_graph = drug_graph

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return self.data_df.iloc[index]

    def _create_b_graph(self,edge_index,x_s, x_t):
        return BipartiteData(edge_index,x_s,x_t)

    def collate_fn(self, batch):
        head_list = []
        tail_list = []
        label_list = []
        rel_list = []
        bgraph_list = []

        head_num = 0
        tail_num = 0
        b_num = 0

        for row in batch:
            Drug1_ID, Drug2_ID, Y, Neg_samples = row['Drug1_ID'], row['Drug2_ID'], row['Y'], row['Neg samples']
            Neg_ID, Ntype = Neg_samples.split('$')
            h_graph = self.drug_graph.get(Drug1_ID)
            t_graph = self.drug_graph.get(Drug2_ID)
            n_graph = self.drug_graph.get(Neg_ID)

            pos_pair_h = h_graph
            pos_pair_t = t_graph

            if Ntype == 'h':
                neg_pair_h = n_graph
                neg_pair_t = t_graph
            else:
                neg_pair_h = h_graph
                neg_pair_t = n_graph  

            head_list.append(pos_pair_h)
            head_list.append(neg_pair_h)
            tail_list.append(pos_pair_t)
            tail_list.append(neg_pair_t)
            bgraph_list.append(self._create_b_graph(get_bipartite_graph(pos_pair_h,pos_pair_t),pos_pair_h.x,pos_pair_t.x))
            bgraph_list.append(self._create_b_graph(get_bipartite_graph(neg_pair_h,neg_pair_t),neg_pair_h.x,neg_pair_t.x))
            head_num += pos_pair_h.num_nodes
            head_num += neg_pair_h.num_nodes
            tail_num += pos_pair_t.num_nodes
            tail_num += neg_pair_t.num_nodes
            b_num += (pos_pair_h.num_nodes + neg_pair_h.num_nodes + pos_pair_t.num_nodes + neg_pair_t.num_nodes)

            rel_list.append(torch.LongTensor([Y]))
            rel_list.append(torch.LongTensor([Y]))

            label_list.append(torch.FloatTensor([1]))
            label_list.append(torch.FloatTensor([0]))

        head_pairs = Batch.from_data_list(head_list, follow_batch=['edge_index'])
        tail_pairs = Batch.from_data_list(tail_list, follow_batch=['edge_index'])
        rel = torch.cat(rel_list, dim=0)
        label = torch.cat(label_list, dim=0)
        bgraphs = Batch.from_data_list(bgraph_list)
        #print(head_num, tail_num, b_num)

        return head_pairs, tail_pairs, rel, label, bgraphs

class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


def split_train_valid(data_df, fold, val_ratio=0.2):
        cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
        train_index, val_index = next(iter(cv_split.split(X=range(len(data_df)), y = data_df['Y'])))

        train_df = data_df.iloc[train_index]
        val_df = data_df.iloc[val_index]

        return train_df, val_df

def load_ddi_dataset(root, batch_size, fold=0):
    drug_graph = read_pickle(os.path.join(root, 'drug_data.pkl'))

    train_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets_train_fold{fold}.csv'))
    test_df = pd.read_csv(os.path.join(root, f'pair_pos_neg_triplets_test_fold{fold}.csv'))
    train_df, val_df = split_train_valid(train_df, fold=fold)   #划分为训练集和验证集

    train_set = DrugDataset(train_df, drug_graph)
    val_set = DrugDataset(val_df, drug_graph)
    test_set = DrugDataset(test_df, drug_graph) 
    train_loader = DrugDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_loader = DrugDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DrugDataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))
        
    return train_loader, val_loader, test_loader

if __name__ == "__main__":

    train_loader, val_loader, test_loader = load_ddi_dataset(root='data/preprocessed/drugbank', batch_size=256, fold=0)


# %%
