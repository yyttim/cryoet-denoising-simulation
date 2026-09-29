import torch
import torch.nn as nn

from collections import OrderedDict
from torch.autograd import Variable

import math
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.utils import spectral_norm




class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])
        
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.training:
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x + epsilon
        else:
            return x



class SpatialGraphConvolution(nn.Module):
    """
    Spatial reasoning layer, similar to https://zero-lab-pku.github.io/publication/2020/cvpr20_spatial_pyramid_graph_reasoning_for_semantic_segmentation/
    """
    def __init__(self, in_features, out_features, attention_dim, bias=True):
        super(SpatialGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.attention_dim = attention_dim

        self.conv =  nn.Conv3d(in_channels=in_features,
                            out_channels=attention_dim,
                            kernel_size=1)

        self.mlp = nn.Linear(in_features, attention_dim, bias=True)

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        N, C, H, W, D = input.shape 

        Phi_T = nn.ReLU()(self.conv(input)).view(N, self.attention_dim, H*W*D)
        Lambda = torch.diag_embed(nn.ReLU()(self.mlp(nn.AvgPool3d(kernel_size=H)(input).view(N, C))))

        ones = torch.ones((N, H*W*D, 1)).to(Phi_T.device)


        M = 1/(torch.sqrt(torch.bmm(Phi_T.permute(0,2,1), torch.bmm(Lambda, torch.bmm(Phi_T, ones))) + 1e-23))

        LX = input.view(N, C, H*W*D).permute(0,2,1)
        LX = LX - torch.bmm(Phi_T.permute(0,2,1) * M, torch.bmm(Lambda, torch.bmm(Phi_T * M.permute(0,2,1), LX)))

        output = torch.bmm(LX, torch.tile(self.weight, (N, 1, 1))).view(N, H, W, D, self.out_features).permute(0,4,1,2,3)

        if self.bias is not None:
            return nn.ELU()(output) + torch.tile(self.bias, (N, H, W, D, 1)).permute(0, 4, 1, 2, 3)
        else:
            return nn.ELU()(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class DilatedConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(DilatedConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features//4

        self.conv1 =  nn.Conv3d(in_channels=self.in_features,
                            out_channels=self.out_features,
                            kernel_size=3,
                            dilation=1,
                            padding=1,
                            bias=True)

        self.conv2 =  nn.Conv3d(in_channels=self.in_features,
                            out_channels=self.out_features,
                            kernel_size=3,
                            dilation=2,
                            padding=2,
                            bias=True)


        self.conv3 =  nn.Conv3d(in_channels=self.in_features,
                            out_channels=self.out_features,
                            kernel_size=3,
                            dilation=3,
                            padding=3,
                            bias=True)

        self.conv4 =  nn.Conv3d(in_channels=self.in_features,
                            out_channels=self.out_features,
                            kernel_size=3,
                            dilation=4,
                            padding=4,
                            bias=True)


    def forward(self, input): 

        return nn.ELU()(torch.cat((self.conv1(input), self.conv2(input), self.conv3(input), self.conv4(input)), dim = 1)) 





class ResidualConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.conv1 =  spectral_norm(nn.Conv3d(in_channels=self.in_features,
                            out_channels=self.out_features,
                            kernel_size=3,
                            dilation=1,
                            padding=1,
                            bias=True))


        self.conv2 =  spectral_norm(nn.Conv3d(in_channels=self.out_features,
                            out_channels=self.out_features,
                            kernel_size=3,
                            dilation=1,
                            padding=1,
                            bias=True))


        self.conv3 =  spectral_norm(nn.Conv3d(in_channels=self.in_features,
                            out_channels=self.out_features,
                            kernel_size=1,
                            dilation=1,
                            padding=0,
                            bias=True))



    def forward(self, input): 

        return nn.ELU()(self.conv2(nn.ELU()(self.conv1(input)))  + self.conv3(input))






'''
class Multiscale_discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=8):
        super(Multiscale_discriminator, self).__init__()

        features = init_features
        self.encoder1 = Multiscale_discriminator._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = Multiscale_discriminator._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = Multiscale_discriminator._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = Multiscale_discriminator._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = Multiscale_discriminator._block(features * 8, features * 16, name="bottleneck")

#        self.global_pool = nn.AvgPool3d(kernel_size=12, stride=12)

        self.flatten = nn.Flatten(start_dim=1)
        
        self.linear1 = spectral_norm(nn.Linear(features * 16 * 12 ** 3, 16))
        self.linear2 = spectral_norm(nn.Linear(16, 1))



    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        vector = self.linear1(self.flatten(bottleneck))
        vector = self.linear2(nn.ELU(inplace=True)(vector))

#        dec4 = self.upconv4(bottleneck)
#        dec4 = torch.cat((dec4, enc4), dim=1)
#        dec4 = self.decoder4(dec4)
#        dec3 = self.upconv3(dec4)
#        dec3 = torch.cat((dec3, enc3), dim=1)
#        dec3 = self.decoder3(dec3)
#        dec2 = self.upconv2(dec3)
#        dec2 = torch.cat((dec2, enc2), dim=1)
#        dec2 = self.decoder2(dec2)
#        dec1 = self.upconv1(dec2)
#        dec1 = torch.cat((dec1, enc1), dim=1)
#        dec1 = self.decoder1(dec1)

        return vector


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        spectral_norm(nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        )),
                    ),
#                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ELU(inplace=True)),
                    (
                        name + "conv2",
                        spectral_norm(nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        )),
                    ),
#                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ELU(inplace=True)),
                ]
            )
        )

    def _spatial_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        SpatialGraphConvolution(in_features = in_channels, out_features = features, attention_dim = features//4),
                    ),
                    (
                        name + "conv2",
                        SpatialGraphConvolution(in_features = features, out_features = features, attention_dim = features//4),
                    ),
                ]
            )
        )


    def _dilated_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        DilatedConvolution(in_features = in_channels, out_features = features),
                    ),
                    (
                        name + "conv2",
                        DilatedConvolution(in_features = features, out_features = features),
                    ),
                ]
            )
        )
'''




class Multiscale_discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(Multiscale_discriminator, self).__init__()

        features = init_features
        self.encoder1 = Multiscale_discriminator._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = Multiscale_discriminator._block(features, features * 2, name="enc2")
#        self.SGC2 = Multiscale_discriminator._spatial_block(features * 2, features * 2, name="sgc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = Multiscale_discriminator._block(features * 2, features * 4, name="enc3")
#        self.SGC3 = Multiscale_discriminator._spatial_block(features * 4, features * 4, name="sgc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = Multiscale_discriminator._block(features * 4, features * 8, name="enc4")
#        self.SGC4 = Multiscale_discriminator._spatial_block(features * 8, features * 8, name="sgc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = Multiscale_discriminator._block(features * 8, features * 16, name="bottleneck")

        self.global_pool1 = nn.MaxPool3d(kernel_size=16, stride=16)
        self.global_pool2 = nn.MaxPool3d(kernel_size=8, stride=8)
        self.global_pool3 = nn.MaxPool3d(kernel_size=4, stride=4)
        self.global_pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.global_pool5 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten(start_dim=1)
        
#        self.linear1 = spectral_norm(nn.Linear(features * (1+2+4+8+16) * 6 ** 3, 64))
        self.linear1 = spectral_norm(nn.Linear(features * 16 * 4 ** 3, 64))
        self.linear2 = spectral_norm(nn.Linear(64, 1))



    def forward(self, x):
        enc1 = self.encoder1(x) 
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        vector = self.linear1(self.flatten(bottleneck))
        vector = self.linear2(nn.ELU(inplace=True)(vector))


#        dec4 = self.upconv4(bottleneck)
#        dec4 = torch.cat((dec4, enc4), dim=1)
#        dec4 = self.decoder4(dec4)
#        dec3 = self.upconv3(dec4)
#        dec3 = torch.cat((dec3, enc3), dim=1)
#        dec3 = self.decoder3(dec3)
#        dec2 = self.upconv2(dec3)
#        dec2 = torch.cat((dec2, enc2), dim=1)
#        dec2 = self.decoder2(dec2)
#        dec1 = self.upconv1(dec2)
#        dec1 = torch.cat((dec1, enc1), dim=1)

#        dec1 = self.decoder1(dec1)

        return vector



    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (

                        name + "conv1",
                        spectral_norm(nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,

                            padding=1,
                            bias=True,
                        )),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),

                    (name + "relu1", nn.ELU(inplace=True)),
                    (
                        name + "conv2",
                        spectral_norm(nn.Conv3d(
                            in_channels=features,
                            out_channels=features,

                            kernel_size=3,
                            padding=1,
                            bias=True,
                        )),
                    ),

                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ELU(inplace=True)),
                ]
            )
        )


    def _spatial_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (

                        name + "conv1",
                        SpatialGraphConvolution(in_features = in_channels, out_features = features, attention_dim = features//4),
                    ),
                    (
                        name + "conv2",

                        SpatialGraphConvolution(in_features = features, out_features = features, attention_dim = features//4),
                    ),
                ]
            )
        )



    def _dilated_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [

                    (
                        name + "conv1",
                        DilatedConvolution(in_features = in_channels, out_features = features),
                    ),
                    (

                        name + "conv2",
                        DilatedConvolution(in_features = features, out_features = features),
                    ),
                ]
            )
        )



    def _residual_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        ResidualConvolution(in_features = in_channels, out_features = features),
                    ),
                ]
            )
        )







class UNet_discriminator(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=4):
        super(UNet_discriminator, self).__init__()

        self.discriminator1 = Multiscale_discriminator(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
        self.discriminator2 = Multiscale_discriminator(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
        self.discriminator3 = Multiscale_discriminator(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
        self.discriminator4 = Multiscale_discriminator(in_channels=in_channels, out_channels=out_channels, init_features=init_features)



    def forward(self, x):

        vector1 = self.discriminator1(x)
        vector2 = self.discriminator2(x)
        vector3 = self.discriminator3(x)
        vector4 = self.discriminator4(x)



        return torch.cat([vector1, vector2, vector3, vector4], dim = 0)      








class UNet_generator(nn.Module):

    def __init__(self, in_channels=34, out_channels=33, init_features=16):
        super(UNet_generator, self).__init__()

        features = init_features
        self.encoder1 = UNet_generator._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet_generator._block(features, features * 2, name="enc2")
#        self.SGC2 = UNet_generator._spatial_block(features * 2, features * 2, name="sgc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet_generator._block(features * 2, features * 4, name="enc3")
#        self.SGC3 = UNet_generator._spatial_block(features * 4, features * 4, name="sgc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet_generator._block(features * 4, features * 8, name="enc4")
#        self.SGC4 = UNet_generator._spatial_block(features * 8, features * 8, name="sgc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet_generator._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16 + 32, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet_generator._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet_generator._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet_generator._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet_generator._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)


    def forward(self, x):
        enc1 = self.encoder1(x) 
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = torch.cat((self.bottleneck(self.pool4(enc4)), torch.randn((x.shape[0], 32, 4, 4, 4), device = x.device)), dim = 1)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)      
 

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "relu1", nn.ELU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "relu2", nn.ELU(inplace=True)),
                ]
            )
        )

    def _spatial_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        SpatialGraphConvolution(in_features = in_channels, out_features = features, attention_dim = features//4),
                    ),
                    (
                        name + "conv2",
                        SpatialGraphConvolution(in_features = features, out_features = features, attention_dim = features//4),
                    ),
                ]
            )
        )

    def _dilated_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        DilatedConvolution(in_features = in_channels, out_features = features),
                    ),
                    (
                        name + "conv2",
                        DilatedConvolution(in_features = features, out_features = features),
                    ),
                ]
            )
        ) 


    def _residual_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        ResidualConvolution(in_features = in_channels, out_features = features),
                    ),
                ]
            )
        )



class UNet_segmentor(nn.Module):

    def __init__(self, in_channels=34, out_channels=33, init_features=16):
        super(UNet_segmentor, self).__init__()

        features = init_features
        self.encoder1 = UNet_segmentor._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet_segmentor._block(features, features * 2, name="enc2")
        self.SGC2 = UNet_segmentor._spatial_block(features * 2, features * 2, name="sgc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet_segmentor._block(features * 2, features * 4, name="enc3")
        self.SGC3 = UNet_segmentor._spatial_block(features * 4, features * 4, name="sgc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet_segmentor._block(features * 4, features * 8, name="enc4")
        self.SGC4 = UNet_segmentor._spatial_block(features * 8, features * 8, name="sgc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet_segmentor._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16 + 32, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet_segmentor._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet_segmentor._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet_segmentor._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet_segmentor._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)


    def forward(self, x):
        enc1 = self.encoder1(x) 
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = torch.cat((self.bottleneck(self.pool4(enc4)), torch.randn((1, 32, 12, 12, 12), device = x.device)), dim = 1)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, self.SGC4(enc4)), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, self.SGC3(enc3)), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, self.SGC2(enc2)), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.nn.functional.softmax(self.conv(dec1), dim=1)      
 

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(

                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
#                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ELU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
#                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ELU(inplace=True)),
                ]
            )
        )

    def _spatial_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        SpatialGraphConvolution(in_features = in_channels, out_features = features, attention_dim = features//4),
                    ),
                    (
                        name + "conv2",
                        SpatialGraphConvolution(in_features = features, out_features = features, attention_dim = features//4),
                    ),
                ]
            )
        )

    def _dilated_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        DilatedConvolution(in_features = in_channels, out_features = features),
                    ),
                    (
                        name + "conv2",
                        DilatedConvolution(in_features = features, out_features = features),
                    ),
                ]
            )
        )


    def _residual_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        ResidualConvolution(in_features = in_channels, out_features = features),
                    ),
                ]
            )
        )





class UNet_translator(nn.Module):

    def __init__(self, in_channels=34, out_channels=33, init_features=16):
        super(UNet_translator, self).__init__()

        features = init_features
        self.encoder1 = UNet_translator._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet_translator._block(features, features * 2, name="enc2")
        self.SGC2 = UNet_translator._spatial_block(features * 2, features * 2, name="sgc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet_translator._block(features * 2, features * 4, name="enc3")
        self.SGC3 = UNet_translator._spatial_block(features * 4, features * 4, name="sgc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet_translator._block(features * 4, features * 8, name="enc4")
        self.SGC4 = UNet_translator._spatial_block(features * 8, features * 8, name="sgc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet_translator._spatial_block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16 + 32, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet_translator._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet_translator._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet_translator._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet_translator._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)


    def forward(self, x):
        enc1 = self.encoder1(x) 
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = torch.cat((self.bottleneck(self.pool4(enc4)), torch.randn((1, 32, 12, 12, 12), device = x.device)), dim = 1)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, self.SGC4(enc4)), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, self.SGC3(enc3)), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, self.SGC2(enc2)), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)


        return self.conv(dec1)     
 

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
#                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ELU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
#                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ELU(inplace=True)),
                ]
            )
        )

    def _spatial_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        SpatialGraphConvolution(in_features = in_channels, out_features = features, attention_dim = features//4),
                    ),
                    (
                        name + "conv2",
                        SpatialGraphConvolution(in_features = features, out_features = features, attention_dim = features//4),
                    ),
                ]
            )
        )

    def _dilated_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        DilatedConvolution(in_features = in_channels, out_features = features),
                    ),
                    (
                        name + "conv2",
                        DilatedConvolution(in_features = features, out_features = features),
                    ),
                ]
            )
        )


    def _residual_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        ResidualConvolution(in_features = in_channels, out_features = features),
                    ),
                ]
            )
        )
