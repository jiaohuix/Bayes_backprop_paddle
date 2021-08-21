import os
import paddle
import numpy as np
from .BayesianLeNet import BBBLeNet
from .BayesianAlexNet import BBBAlexNet
from .Bayesian3Conv3FC import BBB3Conv3FC
from .BayesianMlp import BBBMlp
import model as models
from utils import logger

def build_model(conf):
    # 加载模型
    map_dic={'lenet':'BBBLeNet','alex':'BBBAlexNet','3conv3fc':'BBB3Conv3FC','2fc':'BBBMlp'}
    model_name=map_dic[conf['model']['name']]
    size=conf['data']['input_size']
    input_dim=size[0] if model_name!='BBBMlp' else size[1]*size[2] # 卷积的输入通道数,或全连接的输入dim
    model=getattr(models,model_name)(input_dim=input_dim,
                                    num_classes=conf['data']['class_dim'],
                                    priors=conf['model']['priors'],
                                    layer_type=conf['model']['layer_type'])
    logger.info('Prep | Model loaded!')
    # 加载权重
    model_path=os.path.join(conf['model']["save_dir"],conf['model']["load_name"])
    if os.path.exists(model_path):
        model.set_dict(paddle.load(model_path))
        logger.info('Prep | Model weight loaded!')
    return model