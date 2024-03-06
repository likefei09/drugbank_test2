import pandas as pd
import pickle

id_smile_dict = {}
df = pd.read_csv('./drugbank.tab', delimiter='\t')
for no, line in df.iterrows():
    id1, id2 = line['ID1'], line['ID2']
    smiles1, smiles2 = line['X1'], line['X2']
    if not id1 in id_smile_dict.keys():
        id_smile_dict[id1] = smiles1
    if not id2 in id_smile_dict.keys():
        id_smile_dict[id2] = smiles2
with open('./all_smiles.pkl', 'wb') as f:
    pickle.dump(id_smile_dict, f)