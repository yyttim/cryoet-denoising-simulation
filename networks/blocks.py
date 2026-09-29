import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.init as init

class LeakyReLUConv3d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=True):
        super(LeakyReLUConv3d, self).__init__()
        model = []
        model += [nn.ReplicationPad3d(padding)]
        if sn:
            model += [spectral_norm(nn.Conv3d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv3d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if 'norm' == 'Instance':
            model += [nn.BatchNorm3d(n_out, affine=False)]
        model += [nn.ELU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(weights_init)
    
    def forward(self, x):
        return self.model(x)


class ReLUINSConv3d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv3d, self).__init__()
        model = []
        model += [nn.ReplicationPad3d(padding)]
        model += [nn.Conv3d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        model += [nn.BatchNorm3d(n_out, affine=False)]
        model += [nn.ELU()]
        self.model = nn.Sequential(*model)
        self.model.apply(weights_init)
    
    def forward(self, x):
        return self.model(x)


class ReLUINSConvTranspose3d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose3d, self).__init__()
        model = []
        model += [nn.ConvTranspose3d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
        #model += [nn.LayerNorm(n_out)]
        model += [nn.ELU()]
        self.model = nn.Sequential(*model)
        self.model.apply(weights_init)

    def forward(self, x):
        return self.model(x)
    
class INSResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1):
        return [nn.ReplicationPad3d(1), nn.Conv3d(inplanes, out_planes, kernel_size=3, stride=stride)]
  
    def __init__(self, inplanes, planes, stride=1, dropout=0.3):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.BatchNorm3d(planes)]
        model += [nn.ELU()]
        model += self.conv3x3(planes, planes)
        model += [nn.BatchNorm3d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(weights_init)
        
    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out




def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun
