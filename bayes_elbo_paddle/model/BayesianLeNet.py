import paddle.nn as nn
from .layers import BayesianModule,BayesLinear,BayesLinearLRT
from .layers import  BayesConv2d,BayesConv2dLRT
class BBBLeNet(BayesianModule):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self,input_dim,num_classes, params=None, layer_type='lrt'):
        super(BBBLeNet, self).__init__()

        self.num_classes = num_classes
        self.layer_type = layer_type
        self.params = params

        if layer_type=='lrt':
            BBBLinear = BayesLinearLRT
            BBBConv2d = BayesConv2dLRT
        elif layer_type=='bbb':
            BBBLinear = BayesLinear
            BBBConv2d = BayesConv2d
        else:
            raise ValueError("Undefined layer_type")

        self.feature=nn.Sequential(
            BBBConv2d(input_dim, 6, 5, padding=0, bias=True, params=self.params),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
            BBBConv2d(6, 16, 5, padding=0, bias=True, params=self.params),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
        )

        self.classifier=nn.Sequential(
            BBBLinear(5 * 5 * 16, 120, bias=True, params=self.params),
            nn.ReLU(),
            BBBLinear(120, 84, bias=True, params=self.params),
            nn.ReLU(),
            BBBLinear(84, num_classes, bias=True, params=self.params),
        )

    def forward(self,x):
        x=self.feature(x)
        x=x.flatten(1)
        x=self.classifier(x)
        return x