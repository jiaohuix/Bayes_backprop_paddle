import os
import paddle
import paddle.nn as nn
import numpy as np
from tqdm import tqdm
from paddle.optimizer.lr import ReduceOnPlateau
from paddle.nn import functional as F
from utils import logger,draw_process,ExpDecayWithWarmup
from eval import evaluate_model
from utils.minibatch_weighting import minibatch_weight
def prep_model(conf,model):
    strategy=conf['hparas']['learning_strategy']
    warmup_steps, lr_start, lr_peak, lr_decay=strategy['warmup_steps'], eval(strategy['lr_start']),\
                                                      eval(strategy['lr_peak']),strategy['lr_decay']
    weight_decay=strategy['weight_decay']
    # lr_sched=ExpDecayWithWarmup(warmup_steps, lr_start, lr_peak,lr_decay)
    # lr_sched=paddle.optimizer.lr.ExponentialDecay(learning_rate=lr_peak,gamma=lr_decay,verbose=True)
    # lr/=conf['model']['num_ens']
    # lr*=conf['hparas']['batch_size']/128. # 随着bsz增长，学习率线性增长
    # lr_sched=paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=10, verbose=True)
    # lr_sched=paddle.optimizer.lr.MultiStepDecay(learning_rate=lr, milestones=[150,300,450], gamma=0.5, last_epoch=-1, verbose=True)
    criterion = nn.CrossEntropyLoss()
    # lr_sched=LinearDecayWithWarmup(learning_rate=lr,total_steps=7820,warmup=0.25)
    # lr_sched = paddle.optimizer.lr.LinearWarmup(learning_rate=lr, warmup_steps=5000, start_lr=1e-8, end_lr=lr / 4,
    #                                             verbose=True)
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
    lr_sched = ReduceOnPlateau(learning_rate=lr_peak, patience=20, verbose=True)
    optimizer = paddle.optimizer.AdamW(learning_rate=lr_sched,parameters=model.parameters(),grad_clip=clip)
    return criterion,optimizer,lr_sched

def run_one_epoch(model, optimizer, train_loader, num_ens=3):
    model.train()
    train_loss = 0.0
    accs = []

    for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader, 1)):
        optimizer.clear_grad()
        pi_weight = minibatch_weight(batch_idx=batch_idx, num_batches=len(train_loader))
        loss,acc = model.calc_metric(
            inputs=inputs,
            targets=labels,
            n_samples=num_ens, # 参数
            w_complexity=pi_weight
        )

        # 更新
        loss.backward()
        optimizer.step()
        # metric
        accs.append(acc.item())
        train_loss += loss.item()
    return train_loss/len(train_loader), np.mean(accs)

def train_model(conf,model,train_loader,valid_loader):
    _,optimizer,lr_sched=prep_model(conf,model)
    # Hyper Parameter settings
    n_epochs = conf['hparas']['num_epochs']
    save_dir=conf['model']['save_dir']
    train_ens = conf['model']['train_ens']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    train_losses,val_losses=[],[]
    train_accs,val_accs=[],[]
    train_err,val_err=[],[]
    # valid_acc_max = -np.Inf
    valid_acc_max = 0.9802
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        # train
        train_loss, train_acc = run_one_epoch(model, optimizer, train_loader,num_ens=train_ens)
        # evaluate
        valid_loss, valid_acc = evaluate_model(conf,model,valid_loader,num_batches=len(train_loader))
        lr_sched.step(valid_loss)
        # lr_sched.step()
        logger.info(f'Epoch: {epoch} | Train Loss: {float(train_loss):.4f} | Train Acc: {float(train_acc):.4f} | '
              f'Val Loss: {float(valid_loss):.4f} | Val Acc: {float(valid_acc):.4f} ') # | train_kl_div: {float(train_kl):.4f}

        # save model if validation accuracy has increased
        if valid_acc > valid_acc_max:
            logger.info(f'Val Acc ⬆ ({float(valid_acc_max):.4f} --> {float(valid_acc):.4f}).  Saving model ...')
            paddle.save(model.state_dict(), os.path.join(save_dir,'model_best2.pdparams'))
            valid_acc_max = valid_acc

        if epoch % conf['hparas']['save_epochs'] == 0:
            paddle.save(model.state_dict(), os.path.join(save_dir, f'model_{epoch}.pdparams'))

        # save metrics
        train_losses.append(float(train_loss))
        val_losses.append(float(valid_loss))
        train_accs.append(float(train_acc))
        val_accs.append(float(valid_acc))
        train_err.append(100.-float(train_acc)*100.)
        val_err.append(100.-float(valid_acc)*100.)

        # visualize
        if (epoch+1)%conf['hparas']['visual_epochs']==0:
            draw_process('Train/Val Loss', train_losses, val_losses, 'loss')
            draw_process('Train/Val Acc', train_accs, val_accs, 'acc')
            draw_process('Train/Val Err', train_err, val_err, 'error')
