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

def val(model, criterion, dataloader, device):    #model:模型 criterion:损失函数 dataloader:数据加载其 deveice:指定模型在哪个设备上运行，通常是CPU或GPU
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    for data in dataloader:
        head_pairs, tail_pairs, rel, label, bgraph = [d.to(device) for d in data]

        with torch.no_grad():
            pred = model((head_pairs, tail_pairs, rel, bgraph))
            loss = criterion(pred, label)

            pred_cls = torch.sigmoid(pred)
            pred_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred_probs = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    acc, auroc, f1_score, precision, recall, ap = do_compute_metrics(pred_probs, label)

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss, acc, auroc, f1_score, precision, recall, ap




def main():
    parser = argparse.ArgumentParser()

    # Add argument
    #使用add_argument()方法添加不同的参数。每个参数都有一些选项，如type指定参数的数据类型，default指定参数的默认值，help提供参数的描述等。
    parser.add_argument('--n_iter', type=int, default=10, help='number of iterations')
    parser.add_argument('--fold', type=int, default=0, help='[0, 1, 2]')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    args = parser.parse_args()

    params = dict(
        model='SA-DDI',
        data_root='data/preprocessed/',
        save_dir='save',
        dataset='drugbank',
        epochs=args.epochs,
        fold=args.fold,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        weight_decay=args.weight_decay
    )


    logger = TrainLogger(params)
    logger.info(__file__)

    save_model = params.get('save_model')
    batch_size = params.get('batch_size')
    data_root = params.get('data_root')
    data_set = params.get('dataset')
    fold = params.get('fold')
    epochs = params.get('epochs')
    n_iter = params.get('n_iter')
    lr = params.get('lr')
    weight_decay = params.get('weight_decay')
    data_path = os.path.join(data_root, data_set)

    train_loader, val_loader, test_loader = load_ddi_dataset(root=data_path, batch_size=batch_size, fold=fold)
    data = next(iter(train_loader))
    node_dim = data[0].x.size(-1)
    edge_dim=10   #？？？？为什么直接设置成10了
    print('node dim: ', node_dim)
    device = torch.device('cuda:2')

    #model = SA_DDI(node_dim, edge_dim, n_iter=n_iter).cuda()
    #model = MVN_DDI(in_features=node_dim, hidd_dim=128, rel_total=86, heads_out_feat_params=[64,64,64,64], blocks_params=[2, 2, 2, 2]).to(device)
    model = My_DDI(node_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) #优化器
    criterion = nn.BCEWithLogitsLoss()  #损失函数
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))    #学习率调度器？？？

    running_loss = AverageMeter()  #平均损失
    running_acc = AverageMeter()   #平均准确率


    model.train()  #将模式调整为训练模式
    for epoch in range(epochs):
        for data in tqdm(train_loader):
            #将data中的元素赋给变量，然后移动到指定的设备进行计算
            head_pairs, tail_pairs, rel, label, bgraph = [d.to(device) for d in data]


            pred = model((head_pairs, tail_pairs, rel, bgraph), device)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_cls = (torch.sigmoid(pred) > 0.5).detach().cpu().numpy()
            acc = accuracy(label.detach().cpu().numpy(), pred_cls)
            running_acc.update(acc)
            running_loss.update(loss.item(), label.size(0))

        epoch_loss = running_loss.get_average()
        epoch_acc = running_acc.get_average()
        running_loss.reset()
        running_acc.reset()

        val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap  = val(model, criterion, val_loader, device)

        msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f, val_auroc-%.4f, val_f1_score-%.4f, val_prec-%.4f, val_rec-%.4f, val_ap-%.4f" % (epoch, epoch_loss, epoch_acc, val_loss, val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_ap)
        logger.info(msg)

        scheduler.step()

        if save_model:
            msg = "epoch-%d, train_loss-%.4f, train_acc-%.4f, val_loss-%.4f, val_acc-%.4f" % (epoch, epoch_loss, epoch_acc, val_loss, val_acc)
            # del_file(logger.get_model_dir())
            save_model_dict(model, logger.get_model_dir(), msg)



# %%
if __name__ == "__main__":
    main()

