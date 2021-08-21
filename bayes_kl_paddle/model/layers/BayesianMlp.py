import paddle.nn as nn
from .layers import BBB_Linear
from .layers import BBB_LRT_Linear
from .layers import FlattenLayer, ModuleWrapper


class BBBMlp(ModuleWrapper):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, input_dim, num_classes, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBMlp, self).__init__()

        self.num_classes = num_classes
        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.flatten = FlattenLayer(input_dim)
        self.fc1 = BBBLinear(input_dim, 1200, bias=True, priors=self.priors)
        self.act3 = self.act()

        self.fc2 = BBBLinear(1200, 1200, bias=True, priors=self.priors)
        self.act4 = self.act()

        self.fc3 = BBBLinear(1200, num_classes, bias=True, priors=self.priors)
