import paddle
import numpy as np
import paddle.nn.functional as F
from utils import logmeanexp,metrics
@paddle.no_grad()
def evaluate_model(conf,model,criterion, dev_loader,epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    valid_ens = conf['model']['valid_ens']
    beta_type = conf['model']['beta_type'],

    model.train()
    valid_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(dev_loader):
        outputs = paddle.zeros((inputs.shape[0], model.num_classes, valid_ens))
        kl = 0.0
        for j in range(valid_ens):
            net_out, _kl = model(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, axis=1)
        log_outputs = logmeanexp(outputs, axis=2) # 不同采样的输出求均值然后exp后取log
        beta = metrics.get_beta(i-1, len(dev_loader), beta_type, epoch, num_epochs)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels))

    return valid_loss/len(dev_loader), np.mean(accs)