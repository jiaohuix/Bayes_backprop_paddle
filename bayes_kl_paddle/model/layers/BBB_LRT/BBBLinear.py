import sys
sys.path.append("..")
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from utils.metrics import calculate_kl as KL_DIV
from ..misc import ModuleWrapper
from paddle.nn.initializer import Normal,Constant
normal_=Normal(mean=0.,std=1.)
zeros_=Constant(value=0.)
ones_=Constant(value=1.)

class BBBLinear(ModuleWrapper):

    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        if priors is None:
                priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']
        self.normal=paddle.distribution.Normal(loc=0.,scale=1.) # 正态分布采样eps

        self.W_mu = self.create_parameter(shape=[in_features,out_features],default_initializer=ones_)
        self.W_rho = self.create_parameter(shape=[in_features,out_features],default_initializer=ones_)
        if self.use_bias:
            self.bias_mu = self.create_parameter(shape=[out_features, ], default_initializer=zeros_)
            self.bias_rho = self.create_parameter(shape=[out_features, ], default_initializer=zeros_)
        # else:
        #     self.register_parameter('bias_mu', None)
        #     self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        normal_mu_ = Normal(*self.posterior_mu_initial)
        normal_rho_ = Normal(*self.posterior_rho_initial)
        normal_mu_(self.W_mu)
        normal_rho_(self.W_rho)

        if self.use_bias:
            normal_mu_(self.bias_mu)
            normal_rho_(self.bias_rho)

    def forward(self, x, sample=True):

        self.W_sigma = paddle.log1p(paddle.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = paddle.log1p(paddle.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.linear(x, self.W_mu, self.bias_mu)  # 重参数化trick
        act_var = 1e-16 + F.linear(x ** 2, self.W_sigma ** 2, bias_var)
        act_std = paddle.sqrt(act_var)

        if self.training or sample:
            eps = self.normal.sample(act_mu.shape) # 要修改成从正态分布采样
            return act_mu + act_std * eps
        else:
            return act_mu

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
