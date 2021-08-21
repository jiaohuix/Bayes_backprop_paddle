import paddle
from paddle import Tensor
from .misc import BayesianModule
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform
from model.samplers import GaussianVariational,ScaleMixture


class BayesConv2d(BayesianModule):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, params=None):

        super(BayesConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1

        if params is None:
            params = {
                # 先验P(w)，用mixture得到
                'prior_pi': 0.5,
                'prior_sigma1': 1.0, # -lnσ1=0
                'prior_sigma2': 0.0025, # -lnσ2≈6
                # 近似后验q(w|θ),用来初始化mu和rho
                'posterior_mu_initial': [-0.2, 0.2], # mean std
                'posterior_rho_initial': [-5.0, -4.0],
            }
        # 初始化
        uniform_mu_= Uniform(*params['posterior_mu_initial'])
        uniform_rho_=Uniform(*params['posterior_rho_initial'])

        w_mu = paddle.empty(shape=[out_channels,in_channels,*self.kernel_size])
        w_rho = paddle.empty(shape=[out_channels,in_channels,*self.kernel_size])
        uniform_mu_(w_mu)
        uniform_rho_(w_rho)
        if bias:
            bias_mu = paddle.empty(shape=[out_channels,])
            bias_rho = paddle.empty(shape=[out_channels,])
            uniform_mu_(bias_mu)
            uniform_rho_(bias_rho)

        # 后验概率采样器（P(w|θ)）
        self.w_posterior = GaussianVariational(w_mu, w_rho)
        self.bias_posterior = GaussianVariational(bias_mu, bias_rho)
        # 先验概率采样器 (P(w))
        self.w_prior = ScaleMixture(params['prior_pi'], params['prior_sigma1'], params['prior_sigma2'])
        if bias:
            self.bias_prior = ScaleMixture(params['prior_pi'], params['prior_sigma1'], params['prior_sigma2'])

        self.kl_divergence = 0.0


    def forward(self, x):

        weight = self.w_posterior.sample()
        bias = self.bias_posterior.sample()

        w_log_prior = self.w_prior.log_prior(weight)
        b_log_prior = self.bias_prior.log_prior(bias)

        w_log_posterior = self.w_posterior.log_posterior()
        b_log_posterior = self.bias_posterior.log_posterior()

        total_log_prior = w_log_prior + b_log_prior  # prior =w和b的prior总和
        total_log_posterior = w_log_posterior + b_log_posterior
        self.kl_divergence = self.kld(total_log_prior, total_log_posterior)
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kld(self, log_prior: Tensor, log_posterior: Tensor) -> Tensor:

        """Calculates the KL Divergence. (其实是计算-ELBO，最小化kl散度=最大化ELBO)

        Uses the weight sampled from the posterior distribution to
        calculate the KL Divergence between the prior and posterior.

        Parameters
        ----------
        log_prior : Tensor
            Log likelihood drawn from the prior.
        log_posterior : Tensor
            Log likelihood drawn from the approximate posterior.

        Returns
        -------
        Tensor
            Calculated KL Divergence.
        """

        return log_posterior - log_prior
