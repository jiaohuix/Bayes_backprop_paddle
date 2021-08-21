import os
import random
import paddle
import numpy as np
import matplotlib.pyplot as plt

def same_seeds(seed=2021):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def draw_process(title,train_metric,val_metric,metric_name):
    plt.figure()
    plt.title(title, fontsize=24)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel(metric_name, fontsize=20)
    plt.plot(list(range(len(train_metric))), train_metric,label=f'Train {metric_name}')
    plt.plot(list(range(len(val_metric))), val_metric,label=f'Val {metric_name}')
    plt.legend()
    plt.grid()
    save_dir='image2'
    if not os.path.exists(save_dir):os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,metric_name))
    # plt.show()


def ExpDecayWithWarmup(warmup_steps,lr_start,lr_peak,lr_decay):
    ''' warmup and exponential decay'''
    exp_sched = paddle.optimizer.lr.ExponentialDecay(learning_rate=lr_peak, gamma=lr_decay)
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=exp_sched, warmup_steps=warmup_steps,
                                                 start_lr=lr_start, end_lr=lr_peak, verbose=True)
    return scheduler


def logmeanexp(x, axis=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if axis is None:
        x, axis = x.reshape((-1,)), 0
    x_max= paddle.max(x, axis, keepdim=True)
    x = x_max + paddle.log(paddle.mean(paddle.exp(x - x_max), axis, keepdim=True))
    return x if keepdim else x.squeeze(axis)


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_array_to_file(numpy_array, filename):
    file = open(filename, 'a')
    shape = " ".join(map(str, numpy_array.shape))
    np.savetxt(file, numpy_array.flatten(), newline=" ", fmt="%.3f")
    file.write("\n")
    file.close()


