import paddle
import numpy as np
from utils.minibatch_weighting import minibatch_weight

@paddle.no_grad()
def evaluate_model(conf,model, eval_loader,num_batches=128,robust=True):
    """Calculate ensemble accuracy and NLL Loss"""
    model.eval()
    valid_ens = conf['model']['valid_ens']
    valid_loss = 0.0
    accs = []

    for batch_idx, (inputs, labels) in enumerate(eval_loader,1):
        pi_weight = minibatch_weight(batch_idx=batch_idx, num_batches=num_batches)
        loss, acc = model.calc_metric(
            inputs=inputs,
            targets=labels,
            n_samples=valid_ens,  # 参数
            w_complexity=pi_weight,
            robust=robust
        )
        valid_loss+=loss.item()
        accs.append(acc.item())
    return valid_loss/len(eval_loader), np.mean(accs) # 由于loss差不多大，单是eval的batch少，平均batch loss就很高
