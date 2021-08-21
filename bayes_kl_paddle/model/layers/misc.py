from paddle import nn

class ModuleWrapper(nn.Layer):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children(): #
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value) #这是啥。。

    def forward(self, x):
        for module in self.children():
            x = module(x)

        kl = 0.0
        for layer in self.sublayers(): # 计算含有kl_loss的层的kl
            if hasattr(layer, 'kl_loss'):
                kl = kl + layer.kl_loss()

        return x, kl


class FlattenLayer(ModuleWrapper):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.reshape((-1, self.num_features))
