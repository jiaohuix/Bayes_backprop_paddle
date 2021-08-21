import paddle.nn as nn
from .layers import BayesianModule,BayesLinear,BayesLinearLRT
from .layers import  BayesConv2d,BayesConv2dLRT

class BBB3Conv3FC(BayesianModule):
    """
    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self,input_dim,num_classes, params=None, layer_type='lrt'):
        super(BBB3Conv3FC, self).__init__()

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
            BBBConv2d(input_dim, 32, 5, padding=2, bias=True, params=self.params),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            BBBConv2d(32, 64, 5, padding=2, bias=True, params=self.params),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),
            BBBConv2d(64, 128, 5, padding=1, bias=True, params=self.params),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2)
        )
        self.classifier=nn.Sequential(
            BBBLinear(2 * 2 * 128, 1000, bias=True, params=self.params),
            nn.ReLU(),
            BBBLinear(1000, 1000, bias=True, params=self.params),
            nn.ReLU(),
            BBBLinear(1000, num_classes, bias=True, params=self.params)
        )

    def forward(self,x):
        x=self.feature(x)
        x=x.flatten(1)
        x=self.classifier(x)
        return x