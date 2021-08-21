import paddle
import paddle.nn.functional as F
from utils.metrics import calculate_kl as KL_DIV
from ..misc import ModuleWrapper
from paddle.nn.initializer import Normal,Constant

normal_=Normal(mean=0.,std=1.)
zeros_=Constant(value=0.)
ones_=Constant(value=1.)

class BBBConv2d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, priors=None):

        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
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
        self.W_mu = self.create_parameter(shape=[out_channels, in_channels,*self.kernel_size],default_initializer=zeros_)
        self.W_rho = self.create_parameter(shape=[out_channels, in_channels,*self.kernel_size],default_initializer=zeros_)

        if self.use_bias:
            self.bias_mu = self.create_parameter(shape=[out_channels,], default_initializer=zeros_)
            self.bias_rho = self.create_parameter(shape=[out_channels,], default_initializer=zeros_)
        # else:
        #     self.register_parameter('bias_mu', None)
        #     self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        normal_mu_=Normal(*self.posterior_mu_initial)
        normal_rho_=Normal(*self.posterior_rho_initial)
        normal_mu_(self.W_mu)
        normal_rho_(self.W_rho)

        if self.use_bias:
            normal_mu_(self.bias_mu)
            normal_rho_(self.bias_rho)

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps=self.normal.sample(self.W_mu.shape)
            self.W_sigma = paddle.log1p(paddle.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = self.normal.sample(self.bias_mu.shape)
                self.bias_sigma = paddle.log1p(paddle.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):# 求weight和bias各自q（mu+rho）与先验的kl散度
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
