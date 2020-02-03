import torch
import torchvision.models as models
import torch.nn as nn
from .pretrain_models import resnet_ins101

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

def get_imagenet_model(type, pretrained=True):
    return resnet_ins101(pretrained=pretrained)

class ResNet101(nn.Module):

    def __init__(self, pretrained=True, fc=False):
        super(ResNet101, self).__init__()
        self.base = get_imagenet_model('resnet_ins101', pretrained=pretrained)
        if not fc:
            self.base._modules.pop('fc')

    def forward(self, x):
        return self.forward_full(x, fc=False)

    def forward_full(self, x, fc=False):
        features = []
        # features at each level of layers: 
        # [out_feature, layer1_out_feature, layer2_out_feature, layer3_out_feature, layer4_out_feature, final_out_feature]
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        features.append(x)

        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        features.append(x)

        x = self.base.layer2(x)
        features.append(x)

        x = self.base.layer3(x)
        features.append(x)

        x = self.base.layer4(x)
        features.append(x)
        
        if not fc:
            return features
        x = self.base.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.base.fc(x)
        features.append(x)
        return features

    def forward_partial(self, x, levels=1):
        features = []
        # features at each level of layers:
        # level = [], return number of levels for each feature maps
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        features.append(x)

        if levels == 1:
            return features
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        features.append(x)
        
        if levels == 2:
            return features
        x = self.base.layer2(x)
        features.append(x)

        if levels == 3:
            return features
        x = self.base.layer3(x)
        features.append(x)
        
        if levels == 4:
            return features
        x = self.base.layer4(x)
        features.append(x)
        
        if levels == 5:
            return features
        x = self.base.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.base.fc(x)
        features.append(x)
        return features

    def forward_merge(self, x, merge_feature, merge_layer, fc=False):
        features = []
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        features.append(x)

        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        features.append(x)

        x = torch.cat([x, merge_feature], 1)
        x = merge_layer(x)

        x = self.base.layer2(x)
        features.append(x)
        x = self.base.layer3(x)
        features.append(x)
        x = self.base.layer4(x)
        features.append(x)

        if not fc:
            return features
        x = self.base.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.base.fc(x)
        features.append(x)
        return features
