import paddle.nn as nn
from .layers import BayesianModule,BayesLinear,BayesLinearLRT
from .layers import  BayesConv2d,BayesConv2dLRT

class BBBAlexNet(BayesianModule):
    '''The architecture of AlexNet with Bayesian Layers'''

    def __init__(self,input_dim,num_classes, params=None, layer_type='lrt'):
        super(BBBAlexNet, self).__init__()

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
            BBBConv2d(inputs, 64, 11, stride=4, padding=5, bias=True, params=self.params),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
            BBBConv2d(64, 192, 5, padding=2, bias=True, params=self.params),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
            BBBConv2d(192, 384, 3, padding=1, bias=True, params=self.params),
            nn.ReLU(),
            BBBConv2d(384, 256, 3, padding=1, bias=True, params=self.params),
            nn.ReLU(),
            BBBConv2d(256, 128, 3, padding=1, bias=True, params=self.params),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2)
        )
        self.classifier = BBBLinear(1 * 1 * 128, num_classes, bias=True, params=self.params)

    def forward(self,x):
        x=self.feature(x)
        x=x.flatten(1)
        x=self.classifier(x)
        return x