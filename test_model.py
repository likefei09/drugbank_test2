# %%
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import load_ddi_dataset
from log.train_logger import TrainLogger
from multiscale_model import My_DDI
import argparse
from metrics import *
from utils import *

#忽略警告信息
import warnings
warnings.filterwarnings("ignore")


def main():    #model:模型 criterion:损失函数 dataloader:数据加载其 deveice:指定模型在哪个设备上运行，通常是CPU或GPU
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_iter', type=int, default=10, help='number of iterations')   #迭代次数
    parser.add_argument('--fold', type=int, default=0, help='[0, 1, 2]')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    # parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--model_path', type=str, default='/mnt/sdb/home/lkf/code/drugbank/save/20231104_222108_drugbank/model/epoch-167, train_loss-0.0284, train_acc-0.9943, val_loss-0.1403, val_acc-0.9603.pt', help='learning rate')

    args = parser.parse_args()

    params = dict(
        model='SA-DDI',
        data_root='data/preprocessed/',
        save_dir='save',
        dataset='drugbank',
        epochs=args.epochs,
        fold=args.fold,
        save_model=False,
        lr=args.lr,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        weight_decay=args.weight_decay,
        model_path=args.model_path
    )


    logger = TrainLogger(params)
    logger.info(__file__)

    batch_size = params.get('batch_size')
    data_root = params.get('data_root')
    data_set = params.get('dataset')
    fold = params.get('fold')
    data_path = os.path.join(data_root, data_set)
    model_path = params.get('model_path')
    print(f'Loading model from {model_path}')

    train_loader, val_loader, test_loader = load_ddi_dataset(root=data_path, batch_size=batch_size, fold=fold)
    data = next(iter(train_loader))
    node_dim = data[0].x.size(-1)
    print('node dim: ', node_dim)
    device = torch.device('cuda:1')
    model = My_DDI(node_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in tqdm(test_loader):
        head_pairs, tail_pairs, rel, label, bgraph = [d.to(device) for d in data]

        with torch.no_grad():
            pred = model((head_pairs, tail_pairs, rel, bgraph))
            # loss = criterion(pred, label)

            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            # running_loss.update(loss.item(), label.size(0))

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

    print(f'acc: {acc}, auroc: {auroc}, f1_score: {f1_score}, precision: {precision}, recall: {recall}, ap: {ap}')

    # epoch_loss = running_loss.get_average()
    # running_loss.reset()

    # model.train()

    return acc, auroc, f1_score, precision, recall, ap


if __name__ == '__main__':
    main()