import os
import utils
import paddle
import numpy as np
from tqdm import tqdm
from eval import evaluate_model
from paddle.nn import functional as F
from paddle.optimizer.lr import ReduceOnPlateau
from utils import logger,draw_process,metrics


def prep_model(conf,model,trainset_num):
    strategy=conf['hparas']['learning_strategy']
    lr_start,lr_decay=eval(strategy['lr_start']),eval(strategy['lr_peak']),strategy['lr_decay']
    weight_decay=strategy['weight_decay']
    criterion = metrics.ELBO(trainset_num)
    lr_sched = ReduceOnPlateau(learning_rate=lr_peak,patience=10,factor=0.8,verbose=True)
    optimizer = paddle.optimizer.AdamW(learning_rate=lr_sched,parameters=model.parameters(),weight_decay=weight_decay)
    return criterion,optimizer,lr_sched



def run_one_epoch(model, optimizer, criterion, train_loader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    model.train()
    train_loss = 0.0
    accs = []
    kl_list = []

    for i, (inputs, labels) in tqdm(enumerate(train_loader, 1)):
        optimizer.clear_grad()
        outputs = []
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = model(inputs)
            kl += _kl
            outputs.append(F.log_softmax(net_out, axis=1).unsqueeze(2))
        kl = kl / num_ens  # 多个集成模型，求平均kl （期望）
        kl_list.append(kl.item())
        outputs = paddle.concat(outputs, axis=-1)
        log_outputs = utils.logmeanexp(outputs, axis=2)

        beta = metrics.get_beta(i - 1, len(train_loader), beta_type, epoch, num_epochs)  # 计算kl加权Π
        loss = criterion(log_outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(log_outputs, labels))
        train_loss += loss.numpy()
    return train_loss / len(train_loader), np.mean(accs), np.mean(kl_list)

def train_model(conf,model,train_loader,valid_loader):
    criterion,optimizer,lr_sched=prep_model(conf,model,trainset_num=len(train_loader.dataset))
    # Hyper Parameter settings
    start_epoch = conf['hparas']['start_epoch']
    n_epochs = conf['hparas']['num_epochs']
    save_dir=conf['model']['save_dir']
    train_ens = conf['model']['train_ens']
    beta_type=conf['model']['beta_type']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    train_losses,val_losses=[],[]
    train_accs,val_accs=[],[]
    train_err,val_err=[],[]
    valid_acc_max = -np.Inf
    # valid_acc_max = 0.9858
    for epoch in range(start_epoch,n_epochs):  # loop over the dataset multiple times
        # train
        train_loss, train_acc, train_kl= run_one_epoch(model, optimizer,criterion ,train_loader,num_ens=train_ens,beta_type=beta_type,epoch=epoch,num_epochs=n_epochs)
        # evaluate
        valid_loss, valid_acc = evaluate_model(conf,model, criterion, valid_loader,epoch=epoch, num_epochs=n_epochs)
        lr_sched.step(valid_loss)
        # lr_sched.step()
        logger.info(f'Epoch: {epoch} | Train Loss: {float(train_loss):.4f} | Train Acc: {float(train_acc):.4f} | '
              f'Val Loss: {float(valid_loss):.4f} | Val Acc: {float(valid_acc):.4f} | train_kl_div: {float(train_kl):.4f}')

        # save model if validation accuracy has increased
        if valid_acc > valid_acc_max:
            logger.info(f'Val Acc ⬆ ({float(valid_acc_max):.4f} --> {float(valid_acc):.4f}).  Saving model ...')
            paddle.save(model.state_dict(), os.path.join(save_dir,'model_best.pdparams'))
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
