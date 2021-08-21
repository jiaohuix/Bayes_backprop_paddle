import paddle.nn.functional as F
from .layers import BayesLinear,BayesianModule,BayesLinearLRT
from utils.variational_approximator import variational_approximator

@variational_approximator
class BBBMlp(BayesianModule):
    '''The architecture of 2 hidden layer (1200units) mlp with Bayesian Layers'''
    def __init__(self,input_dim,num_classes,params=None, layer_type='bbb'):
        super(BBBMlp, self).__init__()
        if layer_type=='bbb':
            BBBLinear = BayesLinear
        elif layer_type=='lrt':
            BBBLinear = BayesLinearLRT
        else:
            raise ValueError("Undefined layer_type")

        self.fc1 = BBBLinear(input_dim, 1200, bias=True, params=params)
        self.fc2 = BBBLinear(1200, 1200, bias=True, params=params)
        self.fc3 = BBBLinear(1200, num_classes, bias=True, params=params)

    def forward(self,x):
        bsz=x.shape[0]
        x=x.reshape((bsz,-1))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.softmax(self.fc3(x),axis=-1)
        return x