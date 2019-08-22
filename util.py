import random
import numpy as np
import torch
import os


def seed_everything(seed):
    # seed for everything
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs):
    """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
    lr = lr_init
    if epoch == num_epochs//2 or epoch == num_epochs*3//4:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def mkdir_if_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

