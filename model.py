import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()        
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.leakyreluA(self.convA( torch.cat([up_x, concat_with], dim=1) ) ) )  )


class Decoder(nn.Module):
    def __init__(self, num_features=1280, decoder_width = .6):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)
        
        self.up0 = UpSample(skip_input=features//1 + 320, output_features=features//2)
        self.up1 = UpSample(skip_input=features//2 + 160, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 64, output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 32, output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 24, output_features=features//8)
        self.up5 = UpSample(skip_input=features//8 + 16, output_features=features//16)

        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4,x_block5,x_block6 = features[2], features[4], features[6], features[9], features[15],features[18],features[19]
        x_d0 = self.conv2(x_block6)
        x_d1 = self.up0(x_d0, x_block5)
        x_d2 = self.up1(x_d1, x_block4)
        x_d3 = self.up2(x_d2, x_block3)
        x_d4 = self.up3(x_d3, x_block2)
        x_d5 = self.up4(x_d4, x_block1)
        x_d6 = self.up5(x_d5, x_block0)
        return self.conv3(x_d6)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()       
        self.original_model = models.mobilenet_v2( pretrained=True )

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items(): features.append( v(features[-1]) )
        return features
    
    
def conv_norm_lrelu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
        norm_layer(out_dim), nn.LeakyReLU(0.2,True))
    
    
class NLDis(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(NLDis, self).__init__()
        dis_model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2,
                                               norm_layer= norm_layer, padding=1, bias=use_bias)]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        dis_model += [conv_norm_lrelu(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1,
                                               norm_layer= norm_layer, padding=1, bias=use_bias)]

        

        self.dis_model = nn.Sequential(*dis_model)

        self.end = nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, input):
        output0 = self.dis_model(input)
        output = self.end(output0)
        return output

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder( self.encoder(x) )