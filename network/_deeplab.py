import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.head = nn.ModuleList(
            [
                 nn.Sequential(
                    nn.Conv2d(304, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, c, 1)
                ) for c in num_classes]
        )
        
        self._init_weight()

    def forward(self, feature):
        # print('feature:',list(feature.keys())) # deeplbv3plus feature has 2 keys: feature['low_level'], feature['out']
        # assert False
        low_level_feature = self.project( feature['low_level'] ) # [b/n_gpu, 48, 129, 129]
        output_feature = self.aspp(feature['out'])               # [b/n_gpu, 256, 33, 33]
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False) #[b/n_gpu, 256, 129, 129]
        output_feature = torch.cat( [ low_level_feature, output_feature ], dim=1 ) #[b/n_gpu, 304, 129, 129]

        heads = [h(output_feature) for h in self.head] # len=3
        heads = torch.cat(heads, dim=1)  #[b/n_gpu, curr_nb_classes, 129, 129]

        return heads
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)
                #nn.init.normal_(m.weight, mean=5, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _head_initialize(self):
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.aspp = ASPP(in_channels, aspp_dilate)
        ###
        self.aspp_l1 = ASPP(256, aspp_dilate)
        self.aspp_l2 = ASPP(512, aspp_dilate)
        self.aspp_l3 = ASPP(1024, aspp_dilate)
        ###
        
        self.head = nn.ModuleList(
            [
                 nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, c, 1)
                ) for c in num_classes]
        )
        
        self._init_weight()

    def forward(self, feature):
        '''
            # feature['low_layer_1']: torch.Size([6, 256, 129, 129]) # layer1
            # feature['low_layer_2']: torch.Size([6, 512, 65, 65])   # layer2
            # feature['low_layer_3']: torch.Size([6, 1024, 33, 33])  # layer3
            # feature['out']: torch.Size([6, 2048, 33, 33])          # layer4
        '''
        ###===
        feature_low_1 = self.aspp_l1(feature['low_level_1'])         # torch.Size([b/n_gpu, 256, 129, 129])
        # print('feature_low_1:',feature_low_1.size())
        feature_low_2 = self.aspp_l2(feature['low_level_2'])       # torch.Size([b/n_gpu, 256, 65, 65])
        # print('feature_low_2:',feature_low_2.size())
        feature_low_3 = self.aspp_l3(feature['low_level_3'])       # torch.Size([b/n_gpu, 256, 33, 33])
        # print('feature_low_3:',feature_low_3.size())
        feature = self.aspp(feature['out'])                          # torch.Size([b/n_gpu, 256, 33, 33])
        ret_features = {'feature_l1':feature_low_1, 'feature_l2':feature_low_2, 'feature_l3':feature_low_3, 'feature_out':feature}
        ###===
        heads = [h(feature) for h in self.head] 
        heads = torch.cat(heads, dim=1)                              # torch.Size([b/n_gpu, 17, 33, 33])

        # heads_l1 = [h(feature_low_1) for h in self.head] 
        # heads_l1 = torch.cat(heads_l1, dim=1)                        # torch.Size([b/n_gpu, 17, 129, 129])

        return ret_features, heads


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _head_initialize(self):
        for m in self.head:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module