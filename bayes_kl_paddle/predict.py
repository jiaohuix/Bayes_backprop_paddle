import paddle
import numpy as np
from utils import logmeanexp
import paddle.nn.functional as F

@paddle.no_grad()
def predict(conf,model,test_loader):
    res=[]
    model.train()
    n_samples=conf['model']['valid_ens']

    for (inputs, _) in enumerate(test_loader):
        outputs = paddle.zeros((inputs.shape[0], model.num_classes, n_samples))
        for j in range(n_samples):
            net_out, _kl = model(inputs)
            outputs[:, :, j] = F.log_softmax(net_out, axis=1)
        log_outputs = logmeanexp(outputs, axis=2)  # 不同采样的输出probs求均值然后exp后取log
        pred=np.mean(log_outputs.numpy().argmax(axis=1))
        res.append(list(pred))

    return res

