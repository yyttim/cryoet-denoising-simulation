import torch
import torch.nn as nn
import torch.nn.functional as F

class Dns_joint(nn.Module):
    def __init__(self):
        super(Dns_joint, self).__init__()

        encoder_blocks = [
            self.unet_convs(1, 64),
            nn.Sequential(nn.Conv3d(64, 64, 3, 2, 1),
                self.unet_convs(64, 128)),
            nn.Sequential(nn.Conv3d(128, 128, 3, 2, 1),
                self.unet_convs(128, 256)),
            nn.Sequential(nn.Conv3d(256, 256, 3, 2, 1),
                self.unet_convs(256, 512))
            ]
        self.encoder_A = nn.Sequential(*encoder_blocks)
        self.encoder_B = nn.Sequential(*encoder_blocks)

        decoder_blocks = [
            nn.Sequential(self.unet_convs(512, 256), 
                nn.ConvTranspose3d(256, 256, 2, 2, 0)),
            nn.Sequential(self.unet_convs(512, 128), 
                nn.ConvTranspose3d(128, 128, 2, 2, 0)),
            nn.Sequential(self.unet_convs(256, 64), 
                nn.ConvTranspose3d(64, 64, 2, 2, 0)),
            nn.Conv3d(128, 1, 3, 1, 1)
            ]
        self.decoder = nn.Sequential(*decoder_blocks)

    def unet_convs(self, in_channels, out_channels, padding=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, padding),
            nn.BatchNorm3d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding),
            nn.BatchNorm3d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x, y):
        skip_x, skip_y = [], []
        for i in range(len(self.encoder_A)):
            x = self.encoder_A[i](x)
            y = self.encoder_B[i](y)
            skip_x.append(x)
            skip_y.append(y)
        skip_x.reverse()
        skip_y.reverse()

        x = self.decoder[0](skip_x[0])
        y = self.decoder[0](skip_y[0])
        for i in range(1, len(self.decoder)):
            x = torch.cat((skip_x[i], x), 1)
            y = torch.cat((skip_y[i], y), 1)
            x = self.decoder[i](x)
            y = self.decoder[i](y)

        return x, y

    def B_forward(self, y):
        skip_y = []
        for i in range(len(self.encoder_B)):
            y = self.encoder_B[i](y)
            skip_y.append(y)
        skip_y.reverse()

        y = self.decoder[0](skip_y[0])
        for i in range(1, len(self.decoder)):
            y = torch.cat((skip_y[i], y), 1)
            y = self.decoder[i](y)

        return y   




class Dns_seperate(nn.Module):
    def __init__(self):
        super(Dns_seperate, self).__init__()

        encoder_blocks = [
            self.unet_convs(1, 64),
            nn.Sequential(nn.Conv3d(64, 64, 3, 2, 1),
                self.unet_convs(64, 128)),
            nn.Sequential(nn.Conv3d(128, 128, 3, 2, 1),
                self.unet_convs(128, 256)),
            nn.Sequential(nn.Conv3d(256, 256, 3, 2, 1),
                self.unet_convs(256, 512))
            ]
        self.encoder_A = nn.Sequential(*encoder_blocks)
        self.encoder_B = nn.Sequential(*encoder_blocks)

        decoder_blocks = [
            nn.Sequential(self.unet_convs(512, 256), 
                nn.ConvTranspose3d(256, 256, 2, 2, 0)),
            nn.Sequential(self.unet_convs(512, 128), 
                nn.ConvTranspose3d(128, 128, 2, 2, 0)),
            nn.Sequential(self.unet_convs(256, 64), 
                nn.ConvTranspose3d(64, 64, 2, 2, 0)),
            nn.Conv3d(128, 1, 3, 1, 1)
            ]
        self.decoder_A = nn.Sequential(*decoder_blocks)
        self.decoder_B = nn.Sequential(*decoder_blocks)

    def unet_convs(self, in_channels, out_channels, padding=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, padding),
            nn.BatchNorm3d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding),
            nn.BatchNorm3d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x, y):
        skip_x, skip_y = [], []
        for i in range(len(self.encoder_A)):
            x = self.encoder_A[i](x)
            y = self.encoder_B[i](y)
            skip_x.append(x)
            skip_y.append(y)
        skip_x.reverse()
        skip_y.reverse()

        x = self.decoder_A[0](skip_x[0])
        y = self.decoder_B[0](skip_y[0])
        for i in range(1, len(self.decoder)):
            x = torch.cat((skip_x[i], x), 1)
            y = torch.cat((skip_y[i], y), 1)
            x = self.decoder_A[i](x)
            y = self.decoder_B[i](y)

        return x, y

    def B_forward(self, y):
        skip_y = []
        for i in range(len(self.encoder_B)):
            y = self.encoder_B[i](y)
            skip_y.append(y)
        skip_y.reverse()

        y = self.decoder_B[0](skip_y[0])
        for i in range(1, len(self.decoder)):
            y = torch.cat((skip_y[i], y), 1)
            y = self.decoder_B[i](y)

        return y
