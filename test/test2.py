import numpy as np
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--dataset', type=str,  choices=['drugbank', 'twosides'], default='drugbank',
                        help='Dataset to preprocess.')
parser.add_argument('-n', '--neg_ent', type=int, default=1, help='Number of negative samples')
parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the random number generator')
parser.add_argument('-o', '--operation', type=str,  choices=['all', 'generate_triplets', 'drug_data', 'split'], default='all', help='Operation to perform')
parser.add_argument('-t_r', '--test_ratio', type=float, default=0.2)
parser.add_argument('-n_f', '--n_folds', type=int, default=3)

dataset_columns_map = {
    'drugbank': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
    'twosides': ('Drug1_ID', 'Drug2_ID', 'Drug1', 'Drug2', 'New Y'),
}

dataset_file_name_map = {
    'drugbank': ('data/drugbank.tab', '\t'),
    'twosides': ('data/twosides_ge_500.csv', ',')
}
args = parser.parse_args()

args.random_num_gen = np.random.RandomState(0)

print(args.random_num_gen)