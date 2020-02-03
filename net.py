import torch.nn as nn
import torch.nn.functional as f
from model.resnet101 import ResNet101

'''
single object RA-Net reimplementation
'''
class RANet(nn.Module):

    def __init__(self):
        super(RANet, self).__init__()
        # the first ground truth feature which is used to perform corr with other images
        self.base_kernels = None
        self.base_msk = None
        # previous t-1 foreground mask
        self.prev_msk = None

        # feature encoder
        # base resnet 101 encoder
        self.res_101 = ResNet101(pretrained=True)
        # [e3]: [e] + [3], e: encoder, 3: feature coming from the 3rd feature output from ResNet101
        self.e3 = make_single_layer(512, 128, kernel_size=3)
        self.e4 = make_single_layer(1024, 256, kernel_size=3)
        self.e5 = make_single_layer(2048, 512, kernel_size=3)
        self.lf = make_single_layer(128 + 256 + 512, 512)

        # ranking attention module
        self.RAM = RAM()

        # feature decoder
        # adapter for skip connection on the first decoder
        self.ls13 = make_single_layer(512, 32, upscale=1, kernel_size=1)
        self.ls14 = make_single_layer(1024, 16, upscale=2, kernel_size=1)
        self.ls15 = make_single_layer(2048, 16, upscale=4, kernel_size=1)
        # [d2]: [d] + [2], d: decoder, 2: merge feature from the 2nd feature output from ResNet101
        self.d1 = nn.Sequential(
                  make_single_layer(256 + 64 + 1, 256),
                  make_single_layer(256, 256),
                  MultiScaleBlock(256, 128, d=[1,3,6]),
                  ResBlock(128, 64),
                  nn.UpsampleBilinear2d(scale_factor=2)
                )

        # adapter for skip connection for the second decoder
        self.ls22 = make_single_layer(256, 32, upscale=1, kernel_size=1)
        self.ls23 = make_single_layer(512, 16, upscale=2, kernel_size=1)
        self.ls24 = make_single_layer(1024, 16, upscale=4, kernel_size=1)

        self.d2 = nn.Sequential(
                  make_single_layer(128 + 64, 128),
                  make_single_layer(128, 64),
                  MultiScaleBlock(64, 32, d=[1,3,6]),
                  ResBlock(32, 16),
                  nn.UpsampleBilinear2d(scale_factor=2)
                )
        
        # adapter for skip connection for the third decoder
        self.ls31 = make_single_layer(64, 32, upscale=1, kernel_size=1)
        self.ls32 = make_single_layer(256, 16, upscale=2, kernel_size=1)
        self.ls33 = make_single_layer(512, 16, upscale=4, kernel_size=1)

        self.d3 = nn.Sequential(
                  make_single_layer(32 + 64, 64),
                  make_single_layer(64, 32),
                  MultiScaleBlock(32, 16, d=[1,3,6]),
                  ResBlock(16, 8),
                  nn.Conv2d(16, 1, 3, padding=1)
                )

    def set_init_image(self, img, msk):
        features = self.res_101.forward_partial(x, levels=4)
        encoder_features = self.encoder(features)
        self.base_kernels = f.adaptive_avg_pool2d(encoder_features, [15, 27])
        self.base_msk = msk

    def forward(self, x):
        features = self.res_101.forward_partial(x, level=4)
        # features: [n, 512, 28, 28]
        encoder_features = self.encoder(features)
        fcorr, bcorr = self.RAM(encoder_features, self.base_msk, self.base_kernels)
        out, _ = self.decoder(self.prev_mask, fcorr, bcorr, features)
        return out
    
    # RANet feature encoder
    def encoder(self, features):
        f1 = f.normalize(f.max_pool2d(self.l1(features[2])), 2) 
        f2 = f.normalize(self.l2(features[3]))
        f3 = f.normalize(f.upsample(self.l3(features[4]), scale_factor=2, mode='bilinear'))
        out = f.normalize(self.lf(torch.cat([f1, f2, f3], dim=1)))
        return out

    # RANet final decoder
   def decoder(self, msk, fcorr, bcorr, features):
        out1 = torch.cat([self.ls13(features[2]),
            self.ls14(features[3]),
            self.ls15(features[4]),
            fcorr,
            bcorr,
            f.adaptive_avg_pool2d(msk, fcorr.size()[-2::])])
        out1 = self.d1(out1)
        out2 = torch.cat([self.ls22(features[1]),
            self.ls23(features[2]),
            self.ls24(features[3]),
            out1], 1)
        out2 = self.d2(out2)
        out3 = torch.cat([self.ls31(features[0]),
            self.ls32(features[1]),
            self.ls33(features[2]),
            out2], 1)
        out3 = self.d3(out3)
        out = f.sigmoid(out3)
        return out, []

class RAM(nn.Module):

    def __init__(self):
        super(RAM, self).__init__()
        # attention ranking block, takes a correlated feature maps and output a score
        self.alpha = 0.2
        corr_size = (15, 27)
        # two pathways: 1) foreground 2) background
        self.fg = nn.Sequential(
                  nn.UpsampleBilinear2d(scale_factor=2),
                  make_single_layer(256, 256),
                  ResBlock(256, 128, 1),
                  make_single_layer(256, 128)
                )

        self.bg = nn.Sequential(
                  nn.UpsampleBilinear2d(scale_factor=2),
                  make_single_layer(256, 256),
                  ResBlock(256, 128, 1),
                  make_single_layer(256, 128)
                ) 

        self.ranking = nn.Sequential(
                  make_single_layer(15 * 27, 128),
                  ResBlock(128, 32, 2),
                  make_single_layer(128, 1)
                )

    def forward(self, x, fg_mask, base_features):
        '''
        input:
        - base features from the encoder
        - foreground mask
        - features from the encoder
        output:
        - attention ranking on foreground similarity map
        - attention ranking on background similarity map
        '''
        # reshape the mask to be similar to the size of the features
        fmask = f.adaptive_avg_pool2d(fg_mask.detached(), [15, 27])
        # selecting the pixel with greater or equal to 0.9
        bmask = (1 - fmask).ge(0.9).float()
        k, corr = correlation_func(base_features, x)
        # reweighted correlation score with the mask
        corr2fmask = corr * fmask.view(-1, 15 * 27, 1, 1)
        corr2bmask = corr * bmaks.view(-1, 15 * 27, 1, 1)

        # calculating the attention scores
        att_corr2fmask = f.max_pool2d(corr2fmask, 2).permute(0, 2, 3, 1)
        att_corr2bmask = f.max_pool2d(corr2bmask, 2).permute(0, 2, 3, 1)
        sim_map2fmask = (f.relu(self.ranking(att_corr2fmask))) * (fmask != 0).view(-1, 1, 15 * 27) * self.alpha
        sim_map2bmask = (f.relu(self.ranking(att_corr2bmask))) * (bmask != 0).view(-1, 1, 15 * 27) * self.alpha

        # ranking and select/order the useful features
        corr2fmask_size = corr2fmask.size()[2::]
        max_fmask, indices = f.max_pool2d(corr2fmask, return_incides=True)
        max_bmask, indices = f.max_pool2d(corr2bmask, return_incides=True)
        max_fmask = max_fmask.view(-1, 1, 15 * 27) + sim_map2fmask
        max_bmask = max_bmask.view(-1, 1, 15 * 27) + sim_map2bmask
        _, sort_idx = max_fmask.sort(descending=True, dim=2)
        fg_corr = torch.cat([corr_map.index_select(0, ids[0, :256]).unsqueeze(0) for corr_map, ids in zip(corr, sort_idx)])
        _, sort_idx = max_bmask.sort(descending=True, dim=2)
        bg_corr = torch.cat([corr_map.index_select(0, ids[0, :256]).unsqueeze(0) for corr_map, ids in zip(corr, sort_idx)])
        fg_corr = self.fg(fg_corr)
        bg_corr = self.bg(bg_corr)

        return fg_corr, bg_corr


    # reshape the foreground mask to perform dot product with the feature map
    def resize_mask(self, mask, size):
        # size is a tuple of (h, w), e.g. size=(15, 27)
        return nn.functional.adaptive_avg_pool2d(mask, size)

class ResBlock(nn.Module):

    def __init__(self, in_channels, bottle_neck, dilated=1, group=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, bottle_neck, kernel_size=1, bias=False, groups=group),
                nn.InstanceNorm2d(bottle_neck)
                nn.ReLU(inplace=True)
                nn.Conv2d(bottle_neck, bottle_neck, kernel_size=3, padding=1 * dilated, bias=False, dilated=dilated, groups=group)
                nn.InstanceNorm2d(bottle_neck)
                nn.ReLU(inplace=True)
                nn.Conv2d(bottle_neck, in_channels, kernel_size=1, bias=False, groups=group),
                nn.InstanceNorm2d(in_channels)
                )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        return self.relu(out + x)

class MultiScaleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilations=(1,2,4), groups=1):
        super(MultiScaleBlock, self).__init__()
        d1, d2, d3 = dilations
        self.s1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d1, dilated=d1, bias=False, groups=group)
        self.s2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d2, dilated=d2, bias=False, groups=group)
        self.s3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d3, dilated=d3, bias=False, groups=group)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.s1(x) + self.s2(x) + self.s3(x)
        out = self.relu(out)
        return out

def correlate_func(kernels, features):
    # kernel size: 1, in_channels, 15, 27
    size = kernels.size()
    if len(features) == 1:
        # combine the last two dimensions
        k = convert_kernel(kernels)
        corr = correlate(k, features)
        k = k.unsqueeze(0)
    else:
        corr = []
        k = []
        for i in range(len(features)):
            k_i = convert_kernel(kernels[i:i+1])
            corr_i = correlate(k_i, features[i:i+1])
            k_i = k_i.unsqueeze(0)
            k.append(k_i)
            corr.append(corr_i)
        k = torch.cat(k, dim=0)
        corr = torch.cat(corr, dim=0)
    return k, corr

def convert_kernel(feature_map):
    '''
    convert feature_map to kernels for cross-correlation
    '''
    size = feature_map.size()
    # in_channels x w x h
    return feature_map.view(size[1], size[2] * size[3]).transpose(0,1).unsqueeze(2).unsqueeze(3).contiguous()

def correlate(kernel, feature):
    '''
    perform cross-correlation with kernels on the feature map
    '''
    corr = nn.functional.conv2d(feature, kernel.contiguous(), stride=1)
    return corr

def make_single_layer(in_channels, out_channels, upscale=1, kernel_size = 3, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2)
    if upscale == 1:
        return nn.Sequential(
                nn.InstanceNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, groups=groups))
    else:
        return nn.Sequential(
                nn.InstanceNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, groups=groups)),
                nn.UpsamplingBilinear2d(scale_factor=upscale)  
                )
