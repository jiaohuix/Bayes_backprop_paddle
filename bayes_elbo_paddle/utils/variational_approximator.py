import paddle
import paddle.nn as nn
from paddle import Tensor
import paddle.nn.functional as F
from typing import Any, Optional
from model.layers import BayesianModule


def variational_approximator(model: nn.Layer) -> nn.Layer:

    """Adds Variational Inference functionality to a nn.Module.

    Parameters
    ----------
    model : nn.Module
        Model to use variational approximation with.

    Returns
    -------
    model : nn.Module
        Model with additional variational approximation functionality.
    """

    # 注:对于bbb来说计算的是负的elbo，因为最小化q和P(w)的kl散度，即最大化elbo
    def kl_divergence(self) -> Tensor:

        """Calculates the KL Divergence for each BayesianModule.

        The total KL Divergence is calculated by iterating through the
        BayesianModules in the model. KL Divergence for each module is
        calculated as the difference between the log_posterior and the
        log_prior.

        Returns
        -------
        kl : Tensor
            Total KL Divergence.
        """
        kl = 0
        for layer in self.sublayers():
            if isinstance(layer, BayesianModule):
                kl += layer.kl_divergence
        return kl

    # add `kl_divergence` to the model
    setattr(model, 'kl_divergence', kl_divergence)

    def calc_metric(self,
             inputs: Tensor,
             targets: Tensor,
             n_samples: int,
             w_complexity: Optional[float] = 1.0,
             robust:bool=True) -> Tensor:

        """Samples the ELBO loss for a given batch of data.

        The ELBO loss for a given batch of data is the sum of the
        complexity cost and a data-driven cost. Monte Carlo sampling is
        used in order to calculate a representative loss.

        Parameters
        ----------
        inputs : Tensor
            Inputs to the model.
        targets : Tensor
            Target outputs of the model.
        criterion : Any
            Loss function used to calculate data-dependant loss.
        n_samples : int
            Number of samples to use
        w_complexity : float
            Complexity weight multiplier.
        robust: bool
            Weather use mean pred or mean prob to calculate accuracy,wher robust=true,
            will calculate mean pred,and accuracy will be lower but robust
        Returns
        -------
        Tensor
            Value of the ELBO loss for the given data.
        """
        criterion=nn.CrossEntropyLoss(use_softmax=False) # 网络中已用softmax
        loss = 0
        correct =0
        mean_prob=[]
        for sample in range(n_samples):
            # forward
            outputs = self(inputs)
            # loss
            likeli_loss=criterion(outputs,targets)  # likelihood cost
            # loss += F.nll_loss(paddle.log(outputs),targets)
            elbo_loss= self.kl_divergence() * w_complexity # -elbo*Π ，complexity cost
            loss+=likeli_loss+elbo_loss
            # acc
            if robust:
                pred=paddle.argmax(outputs,axis=1).numpy()
                correct+=(pred==targets.numpy().reshape(-1)).sum()
            else:
                # mean prob
                mean_prob.append(F.softmax(outputs, axis=-1).unsqueeze(2))
        # loss
        loss=loss/n_samples
        # acc
        if robust:
            acc=correct/(n_samples*len(targets))
        else:
            # acc
            mean_prob = paddle.concat(mean_prob, axis=-1)
            mean_prob = paddle.mean(mean_prob, axis=-1)
            pred = paddle.argmax(mean_prob, axis=1).numpy()
            correct = (pred == targets.numpy().reshape(-1)).sum()
            acc = correct / len(targets)
        return loss,acc

        # mean_prob=[]
        # for sample in range(n_samples):
        #     # forward
        #     outputs = self(inputs)
        #     # loss
        #     likeli_loss=criterion(outputs,targets)  # likelihood cost
        #     # loss += F.nll_loss(paddle.log(outputs),targets) # 不稳定
        #     elbo_loss= self.kl_divergence() * w_complexity # -elbo*Π ，complexity cost
        #     loss+=likeli_loss+elbo_loss
        #     # mean prob
        #     mean_prob.append(F.softmax(outputs,axis=-1).unsqueeze(2))
        # # acc
        # mean_prob=paddle.concat(mean_prob,axis=-1)
        # mean_prob=paddle.mean(mean_prob,axis=-1)
        # pred=paddle.argmax(mean_prob,axis=1).numpy()
        # correct=(pred==targets.numpy().reshape(-1)).sum()
        # loss=loss/n_samples
        # acc=correct/len(targets)
        # return loss,acc

    # add `elbo` to the model
    setattr(model, 'calc_metric', calc_metric)

    return model
