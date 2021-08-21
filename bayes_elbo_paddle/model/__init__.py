import os
import paddle
import numpy as np
from .BayesianLeNet import BBBLeNet
from .BayesianAlexNet import BBBAlexNet
from .Bayesian3Conv3FC import BBB3Conv3FC
from .Bayesian2Mlp import BBBMlp
from .samplers import *
from .layers import BayesianModule
import model as models
from utils import logger

def build_model(conf):
    # 修改mixure的σ
    fn=lambda x:1/(np.e**x)
    conf['model']['params']['prior_sigma1']=fn(conf['model']['params']['prior_sigma1'])
    conf['model']['params']['prior_sigma2']=fn(conf['model']['params']['prior_sigma2'])
    # 加载模型
    size=conf['data']['input_size']
    input_dim=size[0] if conf['model']['name']!='BBBMlp' else size[1]*size[2] # 卷积的输入通道数,或全连接的输入dim
    model=getattr(models,conf['model']['name'])(input_dim=input_dim,
                                                num_classes=conf['data']['class_dim'],
                                                params=conf['model']['params'],
                                                layer_type=conf['model']['layer_type'])
    logger.info('Prep | Model loaded!')
    # 加载权重
    model_path=os.path.join(conf['model']["save_dir"],conf['model']["load_name"])
    if os.path.exists(model_path):
        model.set_dict(paddle.load(model_path))
        logger.info('Prep | Model weight loaded!')
    return model