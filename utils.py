import os
import torch

#计算和存储最佳值
class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count

#计算和存储平均值和当前值
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg

#张量进行归一化，使其取值范围在0-1之间
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

#保存训练过程中的模型检查点
def save_checkpoint(model, model_dir, epoch, val_loss, val_acc):
    model_path = os.path.join(model_dir, 'epoch:%d-val_loss:%.3f-val_acc:%.3f.model' % (epoch, val_loss, val_acc))
    torch.save(model, model_path)

#加载模型检查点
def load_checkpoint(model_path):
    return torch.load(model_path)

#保存模型状态字典
def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

#加载模型状态字典
def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

#无限循环迭代器
def cycle(iterable):
    while True:
        print("end")
        for x in iterable:
            yield x





