import paddle
import numpy as np
from paddle import nn
import paddle.nn.functional as F

# evidence low bound
class ELBO(nn.Layer):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, beta):
        assert target.stop_gradient #不要梯度
        return F.nll_loss(input, target, reduction='mean') * self.train_size + beta * kl #未做softmax、log的交叉熵

def acc(outputs, targets):
    targets=targets.numpy().reshape((-1,))
    return np.mean(outputs.numpy().argmax(axis=1) == targets)

# q和先验p(w)的kl
def calculate_kl(mu_p, sig_p, mu_q, sig_q):
    kl = 0.5 * (2 * paddle.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

# 对kl加权
def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1) # mini-batch kl resample,对kl加权，使得初期更受先验影响，后期更受数据影响
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta
