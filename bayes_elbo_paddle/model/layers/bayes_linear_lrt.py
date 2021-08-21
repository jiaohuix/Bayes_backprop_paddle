import paddle
from paddle import Tensor
from .misc import BayesianModule
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform,Assign

class BayesLinearLRT(BayesianModule):

    """Bayesian Linear Layer with Local Reparameterisation Trick.

    Implementation of a Bayesian Linear Layer utilising the 'local
    reparameterisation trick' in order to sample directly from the
    activations.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=True,
                 params=None) -> None:

        """Bayesian Linear Layer with Local Reparameterisation Trick.

        Parameters
        ----------
        in_features : int
            Number of features to feed into the layer.
        out_features : int
            Number of features produced by the layer.
        bias: bool
            Is contain bias.
        params : dict
            Prior and posterior init params.
        """
 # Sigma to be used for the normal distribution in the prior.
        super().__init__()
        if params is None:
            params = {
                # 先验P(w)，用mixture得到
                'prior_pi': 0.5,
                'prior_sigma1': 1.0,  # -lnσ1=0
                'prior_sigma2': 0.0025,  # -lnσ2≈6
                # 近似后验q(w|θ),用来初始化mu和rho
                'posterior_mu_initial': [-0.2, 0.2],  # mean std
                'posterior_rho_initial': [-5.0, -4.0],
            }
        self.in_feature = in_features
        self.out_feature = out_features
        self.std_prior = params['prior_sigma1']
        # 初始化
        uniform_mu_ = Uniform(*params['posterior_mu_initial'])
        uniform_rho_ = Uniform(*params['posterior_rho_initial'])

        w_mu = paddle.empty(shape=[in_features, out_features])
        w_rho = paddle.empty(shape=[in_features, out_features])
        uniform_mu_(w_mu)
        uniform_rho_(w_rho)

        self.w_mu = self.create_parameter(shape=w_mu.shape,default_initializer=Assign(w_mu))
        self.w_rho = self.create_parameter(shape=w_rho.shape,default_initializer=Assign(w_rho))
        if bias:
            bias_mu = paddle.empty(shape=[out_features,])
            bias_rho = paddle.empty(shape=[out_features,])
            uniform_mu_(bias_mu)
            uniform_rho_(bias_rho)
            self.bias_mu = self.create_parameter(shape=bias_mu.shape, default_initializer=Assign(bias_mu))
            self.bias_rho = self.create_parameter(shape=bias_rho.shape, default_initializer=Assign(bias_rho))
        self.epsilon_normal = paddle.distribution.Normal(0, 1)
        self.kl_divergence = 0.0

    def forward(self, x: Tensor) -> Tensor:

        """Calculates the forward pass through the linear layer.

        The local reparameterisation trick is used to estimate the
        gradients with respect to the parameters of a distribution - it
        takes advantage of the fact that, for a fixed input and Gaussian
        distributions over the weights, the resulting distribution over
        the activations is also Gaussian.

        Instead of sampling the weights individually and using them to
        compute a sample from the activation - we can sample from the
        distribution over activations. This yields a lower variance
        gradient estimator which makes training faster and more stable.

        Parameters
        ----------
        x : Tensor
            Inputs to the Bayesian Linear Layer.

        Returns
        -------
        Tensor
            Output from the Bayesian Linear Layer.
        """

        w_std = paddle.log(1 + paddle.exp(self.w_rho))
        b_std = paddle.log(1 + paddle.exp(self.bias_rho))

        act_mu = F.linear(x, self.w_mu)
        act_std = paddle.sqrt(F.linear(x.pow(2), w_std.pow(2)))

        w_eps = self.epsilon_normal.sample(act_mu.shape)
        bias_eps = self.epsilon_normal.sample(b_std.shape)

        w_out = act_mu + act_std * w_eps
        b_out = self.bias_mu + b_std * bias_eps

        w_kl = self.kld(
            mu_prior=0.0,
            std_prior=self.std_prior,
            mu_posterior=self.w_mu,
            std_posterior=w_std
        )

        bias_kl = self.kld(
            mu_prior=0.0,
            std_prior=0.1,
            mu_posterior=self.bias_mu,
            std_posterior=b_std
        )

        self.kl_divergence = w_kl + bias_kl

        return w_out + b_out

    def kld(self,
            mu_prior: float,
            std_prior: float,
            mu_posterior: Tensor,
            std_posterior: Tensor) -> Tensor:

        """Calculates the KL Divergence.

        The only 'downside' to the local reparameterisation trick is
        that, as the weights are not being sampled directly, the KL
        Divergence can not be calculated through the use of MC sampling.
        Instead, the closed form of the KL Divergence must be used;
        this restricts the prior and posterior to be Gaussian.

        However, the use of a Gaussian prior / posterior results in a
        lower variance and hence faster convergence.

        Parameters
        ----------
        mu_prior : float
            Mu of the prior normal distribution.
        std_prior : float
            Sigma of the prior normal distribution.
        mu_posterior : Tensor
            Mu to approximate the posterior normal distribution.
        std_posterior : Tensor
            Sigma to approximate the posterior normal distribution.

        Returns
        -------
        Tensor
            Calculated KL Divergence.
        """

        kl_divergence = 0.5 * (
                2 * paddle.log(std_prior / std_posterior) -
                1 +
                (std_posterior / std_prior).pow(2) +
                ((mu_prior - mu_posterior) / std_prior).pow(2)
        ).sum()

        return kl_divergence
